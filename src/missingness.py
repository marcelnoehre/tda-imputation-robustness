from mdatagen.multivariate.mMCAR import mMCAR
from mdatagen.multivariate.mMAR import mMAR
from mdatagen.multivariate.mMNAR import mMNAR
from src.constants import *

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

MISSINGNESS = {
    MCAR: {FUNCTION: lambda df, target, missing_rate, seed: multivariat_mcar(df, target, missing_rate, seed), DETERMINISTIC: False},
    MAR: {FUNCTION: lambda df, target, missing_rate, _: multivariat_mar(df, target, missing_rate), DETERMINISTIC: True},
    MNAR: {FUNCTION: lambda df, target, missing_rate, _: multivariat_mnar(df, target, missing_rate), DETERMINISTIC: True}
}