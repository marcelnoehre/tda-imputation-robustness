import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from src.constants import *

def downstream_score(X, y, dataset_type, seed):
    '''
    Estimate the downstream score of a simple model predicting `y` from `X`.
    '''
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    if n < 2 * DOWNSTREAM_CV_FOLDS:
        return np.nan

    if dataset_type == REGRESSION:
        model = RandomForestRegressor(n_estimators=DOWNSTREAM_ESTIMATORS, random_state=seed, n_jobs=N_JOBS)
        cv = KFold(n_splits=DOWNSTREAM_CV_FOLDS, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    else:
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or counts.min() < 2:
            return np.nan
        folds = min(DOWNSTREAM_CV_FOLDS, int(counts.min()))
        model = RandomForestClassifier(n_estimators=DOWNSTREAM_ESTIMATORS, random_state=seed, n_jobs=N_JOBS)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return float(np.mean(scores))
