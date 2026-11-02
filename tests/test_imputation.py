import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.constants import *
from src.imputation import impute


def _masked():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 5)) * [1, 2, 3, 4, 5] + [0, 1, 2, 3, 4], columns=[f'X{i}' for i in range(5)])
    # mask the lowest values of two columns, so the observed statistics differ from the complete ones
    for j in (1, 3):
        X.loc[X[f'X{j}'].nsmallest(30).index, f'X{j}'] = np.nan
    return X


def test_observed_entries_are_kept():
    X = _masked()
    for imp in (CONSTANT, MEAN, KNN):
        out = impute(X, imp, 1)
        assert np.array_equal(out[X.notna().to_numpy()], X.to_numpy()[X.notna().to_numpy()])


def test_mean_and_constant():
    X = _masked()
    miss = X.isna().to_numpy()
    assert np.allclose(impute(X, MEAN, 1)[miss], np.broadcast_to(X.mean().to_numpy(), X.shape)[miss])
    # Constant is the oracle complete-data mean, 0 in the evaluation scale
    assert np.all(impute(X, CONSTANT, 1)[miss] == 0)


def test_knn_runs_in_the_observed_scale():
    X = _masked()
    mean, std = X.mean().to_numpy(), X.std(ddof=0).to_numpy()
    k = min(int(np.sqrt(len(X))), len(X) // 2)
    expected = KNNImputer(n_neighbors=k).fit_transform((X.to_numpy() - mean) / std) * std + mean
    assert np.allclose(impute(X, KNN, 1), expected)
