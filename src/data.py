import numpy as np
import pandas as pd
import tadasets
import librosa
from sklearn.datasets import fetch_openml

from src.constants import *
from src.utils import numeric_target_mapping

def torus():
    """
    Generate a torus dataset.
    """
    data = tadasets.torus(n=TORUS_SAMPLES, ambient=TORUS_AMBIENT, seed=TORUS_SEED)
    return {DATA: pd.DataFrame(data, columns=[f'X{i}' for i in range(data.shape[1])])}

def on_and_on():
    """
    Generate a dataset from the 'On and On' audio file by computing chroma features
    with sliding windows. The resulting sequence of chroma vectors forms a point cloud
    representing the harmonic structure over time.
    """
    try:
        y, sr = librosa.load(AUDIO_PATH, sr=None)
    except FileNotFoundError:
        y, sr = librosa.load(f'../{AUDIO_PATH}', sr=None)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

    window_size = int(WINDOW_DURATION * sr / HOP_LENGTH)
    step_size = int(STEP_DURATION * sr / HOP_LENGTH)
    windows = np.stack([chroma[:, start:(start + window_size)] 
                        for start in range(0, chroma.shape[1] - window_size + 1, step_size)])

    return {DATA: pd.DataFrame(windows.mean(axis=2), columns=CHROMA_LABELS)}

def get_data(id):
    """
    Fetch a dataset from OpenML by its ID.

    :param int id: The OpenML dataset ID.
    """
    return fetch_openml(data_id=id, as_frame=True)

def preprocess(dataset, key):
    """
    Preprocess the dataset by ensuring it has a target variable and valid column labels.
    
    :param pd.df dataset: The dataset to preprocess.
    :param str key: The key to identify the dataset.

    :return: Preprocessed dataset
    """
    if key == RED_WINE_QUALITY:
        dataset[TARGET] = dataset[DATA].pop('quality')
    if not hasattr(dataset, TARGET) or dataset[TARGET] is None:
        if key in [CNN_STOCK_PRED_DJI, HUNGARIAN_CHICKENPOX]:
            dataset[DATA] = dataset[DATA].drop(columns=['Date'], errors='ignore')
        dataset[TARGET] = np.array(len(dataset[DATA]) * [0])
    else:
        dataset[TARGET] = numeric_target_mapping(dataset[TARGET])

    dataset[DATA].columns = [
        f'X{i}' if not str(c).isidentifier() else str(c).replace(' ', '_') 
        for i, c in enumerate(dataset[DATA].columns)
    ]

    for col in dataset[DATA].select_dtypes(include='int').columns:
        dataset[DATA][col] = dataset[DATA][col].astype(float)

    return dataset
            
def get_all_datasets():
    """
    Fetch and preprocess all datasets.

    :return dict: Dictionary containing all preprocessed datasets.
    """
    return {
        STOCK: preprocess(get_data(STOCK_DATASET_ID), STOCK),
        RMFTSA_LADATA: preprocess(get_data(RMFTSA_LADATA_DATASET_ID), RMFTSA_LADATA),
        CONCRETE_DATA: preprocess(get_data(CONCRETE_DATA_ID), CONCRETE_DATA),
        TREASURY: preprocess(get_data(TREASURY_DATASET_ID), TREASURY),
        WEATHER_IZMIR: preprocess(get_data(WEATHER_IZMIR_DATASET_ID), WEATHER_IZMIR),
        HUNGARIAN_CHICKENPOX: preprocess(get_data(HUNGARIAN_CHICKENPOX_DATASET_ID), HUNGARIAN_CHICKENPOX),
        CNN_STOCK_PRED_DJI: preprocess(get_data(CNN_STOCK_PRED_DJI_DATASET_ID), CNN_STOCK_PRED_DJI),
        DIABETES: preprocess(get_data(DIABETES_DATASET_ID), DIABETES),
        RED_WINE_QUALITY: preprocess(get_data(RED_WINE_QUALITY_DATASET_ID), RED_WINE_QUALITY),
        TORUS: preprocess(torus(), TORUS),
        ON_AND_ON: preprocess(on_and_on(), ON_AND_ON)
    }