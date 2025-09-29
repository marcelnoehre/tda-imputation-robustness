import sys
import os

from pyparsing import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))

import pandas as pd
from src.constants import *

def sort(path):
    df = pd.read_csv(path)

    df[DATASET] = pd.Categorical(df[DATASET], categories=COLLECTIONS[DATA], ordered=True)
    df[MISSINGNESS_TYPE] = pd.Categorical(df[MISSINGNESS_TYPE], categories=COLLECTIONS[MISSINGNESS_TYPE], ordered=True)
    df[IMPUTATION_METHOD] = pd.Categorical(df[IMPUTATION_METHOD], categories=COLLECTIONS[IMPUTATION_METHOD], ordered=True)
    if TDA_METHOD in df.columns:
        df[TDA_METHOD] = pd.Categorical(df[TDA_METHOD], categories=COLLECTIONS[TDA_METHOD], ordered=True)
    if DIMENSION in df.columns:
        df[DIMENSION] = pd.Categorical(df[DIMENSION], categories=DIMENSIONS, ordered=True)

    sort_columns = [DATASET, MISSINGNESS_TYPE, MISSING_RATE, IMPUTATION_METHOD]
    if TDA_METHOD in df.columns:
        sort_columns.append(TDA_METHOD)
    if DIMENSION in df.columns:
        sort_columns.append(DIMENSION)

    df_sorted = df.sort_values(by=sort_columns)
    columns = sort_columns.copy() + [col for col in [MAE, RMSE, WS, BN, L2PL, L2PI] if col in df.columns]
    df_sorted = df_sorted[columns]
    df_sorted.to_csv(path, index=False)

if __name__ == '__main__':
    folder = Path("results")
    csv_files = list(folder.glob("*.csv"))
    for file in csv_files:
        sort(file)
