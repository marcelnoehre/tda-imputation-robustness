import pandas as pd
from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.univariate.uMNAR import uMNAR
from mdatagen.univariate.uMAR import uMAR
from src.utils import *

def univariat_mcar(df, col, target_col, missing_rate, seed=None):
    generator = uMCAR(
        X=df.copy().drop(columns=[target_col]),
        y=map_target_to_np_array(df, target_col),
        missing_rate=missing_rate,
        x_miss=col,
        seed=seed
    )
    return generator.random()

def univariat_mnar(df, col, target_col, threshold):
    generator = uMNAR(
        X=df.copy().drop(columns=[target_col]),
        y=map_target_to_np_array(df, target_col),
        threshold=threshold,
        x_miss=col
    )
    return generator.run()

def univariat_umar(df, col_miss, col_obs, target_col, missing_rate):
    generator = uMAR(
        X=df.copy().drop(columns=[target_col]),
        y=map_target_to_np_array(df, target_col),
        missing_rate=missing_rate,
        x_miss=col_miss,
        x_obs=col_obs
    )
    return generator.mix()