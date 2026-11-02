import warnings
# mdatagen 0.2.0 imports the deprecated pkg_resources when it is imported; harmless, so only this warning is silenced
warnings.filterwarnings('ignore', message='pkg_resources is deprecated', category=UserWarning)
from multiprocessing.pool import ThreadPool
import mdatagen.multivariate.mMAR as _mmar_mod
import mdatagen.multivariate.mMNAR as _mnar_mod
_mmar_mod.Pool = ThreadPool
_mnar_mod.Pool = ThreadPool

from mdatagen.multivariate.mMCAR import mMCAR
from mdatagen.multivariate.mMAR import mMAR
from mdatagen.multivariate.mMNAR import mMNAR
from src.constants import *
from src.config import SEEDS, N_JOBS, MISSING_FEATURE_FRACTION
import random
import numpy as np

def _n_missing_features(df):
    return int(df.shape[1] * MISSING_FEATURE_FRACTION)

def multivariat_mcar(df, target, missing_rate, seed):
    random.seed(seed)
    columns = random.sample(df.columns.tolist(), _n_missing_features(df))
    df_miss = df.copy()
    df_miss[columns] = mMCAR(
        X=df[columns],
        y=target,
        missing_rate=missing_rate * (df.shape[1] / len(columns)),
        seed=seed
    ).random()[columns]
    return df_miss

def multivariat_mar(df, target, missing_rate):
    return mMAR(
        X=df,
        y=target,
        n_xmiss=_n_missing_features(df),
        n_Threads=N_JOBS
    ).correlated(missing_rate)

def multivariat_mnar(df, target, missing_rate, seed):
    # mdatagen draws the masked columns from the global NumPy generator; seed it for this call only
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return mMNAR(
            X=df,
            y=target,
            n_xmiss=_n_missing_features(df),
            n_Threads=N_JOBS
        ).random(missing_rate, deterministic=True).drop(columns=[TARGET])
    finally:
        np.random.set_state(state)

def repeated(seed, mt, method_deterministic):
    """A (mechanism, method) pair runs for every seed as soon as either of them is stochastic.
    Deterministic masks are computed once and shared by all seeds (see introduce_missingness)."""
    return seed == SEEDS[0] or not MISSINGNESS[mt][DETERMINISTIC] or not method_deterministic

MISSINGNESS = {
    MCAR: {FUNCTION: lambda df, target, missing_rate, seed: multivariat_mcar(df, target, missing_rate, seed), DETERMINISTIC: False},
    MAR: {FUNCTION: lambda df, target, missing_rate, _: multivariat_mar(df, target, missing_rate), DETERMINISTIC: True},
    MNAR: {FUNCTION: lambda df, target, missing_rate, seed: multivariat_mnar(df, target, missing_rate, seed), DETERMINISTIC: False}
}