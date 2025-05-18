from src.constants import SEED, MEDIAN
from src.utils import numeric_target_mapping
from mdatagen.multivariate.mMAR import mMAR
from mdatagen.multivariate.mMNAR import mMNAR
from mdatagen.multivariate.mMCAR import mMCAR

def multivariat_mcar(df, target, missing_rate):
    generator = mMCAR(
        X=df,
        y=numeric_target_mapping(target),
        missing_rate=missing_rate,
        seed=SEED
    )
    return generator.random()

def multivariat_nmar(df, target, missing_rate):
    generator = mMNAR(
        X=df,
        y=numeric_target_mapping(target),
        threshold=MEDIAN
    )
    return generator.MBOV_median(missing_rate, df.columns.tolist())

def multivariat_mar(df, target, missing_rate):
    generator = mMAR(
        X=df,
        y=numeric_target_mapping(target),
        n_xmiss=4
    )
    return generator.correlated(missing_rate)
