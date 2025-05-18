from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from statsmodels.imputation.mice import MICEData
from src.constants import SEED

def impute_simple(data, strategy, value=None):
    # Simple (mean, median, most_frequent, constant)
    imp = SimpleImputer(strategy=strategy, fill_value=value)
    return imp.fit_transform(data)

def impute_knn(data, n_neighbors=5):
    # K-Nearest Neighbors Imputation
    knn_imp = KNNImputer(n_neighbors=n_neighbors)
    return knn_imp.fit_transform(data)

def impute_iterative(data):
    # Iterative Imputer (similar to MICE)
    iter_imp = IterativeImputer(random_state=SEED)
    return iter_imp.fit_transform(data)

def impute_mice(data):
    # MICE (Multiple Imputation by Chained Equations)
    mice_imp = MICEData(data)
    mice_imp.update_all()
    return mice_imp.data.values
