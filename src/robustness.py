import numpy as np
from src.constants import SUBSAMPLE_FRACTION

def complete_case_analysis(missing_df):
    '''
    Drop rows with any missing value. Returns (array, retained_row_indices).
    '''
    complete = missing_df.dropna()
    return complete.to_numpy(), complete.index.to_numpy().astype(np.float64)

def random_subsample(dataset_data, seed):
    '''
    Sample SUBSAMPLE_FRACTION of rows, chosen uniformly at random from the
    original clean data. Returns (array, retained_row_indices).
    '''
    dataset_data = np.asarray(dataset_data)
    n_subsample = round(len(dataset_data) * SUBSAMPLE_FRACTION)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset_data), size=n_subsample, replace=False))
    return dataset_data[idx], idx.astype(np.float64)

def shuffled_ground_truth(dataset_data, missing_df, seed):
    '''
    Copy of dataset_data; for each column, the TRUE (pre-missingness)
    values at that column's missing row-positions are permuted among
    themselves and written back at those same positions. Observed cells are
    untouched. Tests whether persistent homology is sensitive to the
    spatial/joint placement of values rather than just their marginal
    distribution per column.
    '''
    dataset_data = np.asarray(dataset_data, dtype=np.float64)
    mask = missing_df.isna().to_numpy()
    out = dataset_data.copy()
    rng = np.random.default_rng(seed)
    for col in range(mask.shape[1]):
        miss_idx = np.where(mask[:, col])[0]
        if len(miss_idx) > 1:
            out[miss_idx, col] = rng.permutation(dataset_data[miss_idx, col])
    return out
