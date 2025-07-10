from mdatagen.multivariate.mMCAR import mMCAR
from mdatagen.multivariate.mMAR import mMAR
from mdatagen.multivariate.mMNAR import mMNAR
from src.constants import *
import random

def multivariat_mcar(df, target, missing_rate, seed):
    random.seed(seed)
    columns = random.sample(df.columns.tolist(), df.shape[1] // 2)
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
        n_xmiss=df.shape[1] // 2,
        n_Threads=N_JOBS
    ).correlated(missing_rate)

def multivariat_mnar(df, target, missing_rate):
    return mMNAR(
        X=df,
        y=target,
        n_xmiss=df.shape[1] // 2,
        n_Threads=N_JOBS
    ).random(missing_rate, deterministic=True).drop(columns=[TARGET])

MISSINGNESS = {
    MCAR: {FUNCTION: lambda df, target, missing_rate, seed: multivariat_mcar(df, target, missing_rate, seed), DETERMINISTIC: False},
    MAR: {FUNCTION: lambda df, target, missing_rate, _: multivariat_mar(df, target, missing_rate), DETERMINISTIC: True},
    MNAR: {FUNCTION: lambda df, target, missing_rate, _: multivariat_mnar(df, target, missing_rate), DETERMINISTIC: True}
}