import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from statsmodels.imputation.mice import MICEData
from src.constants import *

import warnings
warnings.filterwarnings('ignore')

def impute_simple(data, strategy, value=None):
    simple_imp = SimpleImputer(strategy=strategy, fill_value=value)
    return simple_imp.fit_transform(data)

def impute_knn(data):
    k = int(np.sqrt(len(data)))
    knn_imp = KNNImputer(n_neighbors=k)
    return knn_imp.fit_transform(data)

def impute_random_forest(data, seed):
    rf = RandomForestRegressor(
        random_state=seed,
        n_jobs=N_JOBS,
        n_estimators=ESTIMATORS,
        max_depth=DEPTH,
    )
    rf_imp = IterativeImputer(
        estimator=rf,
        max_iter=ITERATIONS,
        random_state=seed
    )
    return rf_imp.fit_transform(data)

def impute_mice(data, seed):
    np.random.seed(seed)
    mice_imp = MICEData(data)
    mice_imp.update_all(n_iter=ITERATIONS)
    return mice_imp.data.values

IMPUTATION_METHODS = {
    'constant': {'fn': lambda data, seed: impute_simple(data, 'constant', 0), 'deterministic': True},
    'mean': {'fn': lambda data, seed: impute_simple(data, 'mean'), 'deterministic': True},
    'median': {'fn': lambda data, seed: impute_simple(data, 'median'), 'deterministic': True},
    'knn': {'fn': lambda data, seed: impute_knn(data), 'deterministic': True},
    'random_forest': {'fn': lambda data, seed: impute_random_forest(data, seed), 'deterministic': False},
    'mice': {'fn': lambda data, seed: impute_mice(data, seed), 'deterministic': False}
}