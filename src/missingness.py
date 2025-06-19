from mdatagen.multivariate.mMCAR import mMCAR
from mdatagen.multivariate.mMAR import mMAR
from mdatagen.multivariate.mMNAR import mMNAR
from src.constants import MEDIAN
from src.constants import N_JOBS

def multivariat_mcar(df, target, missing_rate, seed):
    return mMCAR(
        X=df,
        y=target,
        missing_rate=missing_rate,
        seed=seed
    ).random()

def multivariat_mar(df, target, missing_rate):
    return mMAR(
        X=df,
        y=target,
        n_xmiss=df.shape[1] // 2,
        n_Threads=N_JOBS
    ).correlated(missing_rate)

def multivariat_mnar(df, target, missing_rate):
    return mMNAR(
        X=df,
        y=target,
        threshold=MEDIAN,
        n_xmiss=df.shape[1] // 2,
        n_Threads=N_JOBS
    ).MBOV_median(missing_rate, df.columns.tolist())

MISSINGNESS_METHODS = {
    'MCAR': {'fn': lambda df, target, missing_rate, seed: multivariat_mcar(df, target, missing_rate, seed)},
    'MAR': {'fn': lambda df, target, missing_rate, seed: multivariat_mar(df, target, missing_rate)},
    'MNAR': {'fn': lambda df, target, missing_rate, seed: multivariat_mnar(df, target, missing_rate)}
}