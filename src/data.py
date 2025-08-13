import numpy as np
import pandas as pd
import tadasets
import librosa
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from src.constants import *
from src.utils import numeric_target_mapping

def torus():
    """
    Generate a torus dataset.
    """
    data = tadasets.torus(n=MANIFOLD_SAMPLES, ambient=MANIFOLD_AMBIENT, seed=MANIFOLD_SEED)
    return {DATA: pd.DataFrame(data, columns=[f'X{i}' for i in range(data.shape[1])])}

def swiss_roll():
    """
    Generate a Swiss roll dataset.
    """
    data = tadasets.swiss_roll(n=MANIFOLD_SAMPLES, ambient=MANIFOLD_AMBIENT, seed=MANIFOLD_SEED)
    return {DATA: pd.DataFrame(data, columns=[f'X{i}' for i in range(data.shape[1])])}

def sphere():
    """
    Generate a sphere dataset.
    """
    data = tadasets.sphere(n = MANIFOLD_SAMPLES, ambient=MANIFOLD_AMBIENT, seed=MANIFOLD_SEED)
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
    if id == -1:
        return torus()
    if id == -2:
        return swiss_roll()
    if id == -3:
        return sphere()
    if id == -4:
        return on_and_on()
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
    if key == BOSTON_WEATHER:
        dataset[TARGET] = dataset[DATA].pop('Events')
        dataset[DATA] = dataset[DATA].drop(columns=['Year', 'Month', 'Day'], errors='ignore')
    if key == GENDER_RECOGNITION:
        dataset[TARGET] = dataset[DATA].pop('label')
    if not hasattr(dataset, TARGET) or dataset[TARGET] is None:
        if key in [CNN_STOCK_PRED_DJI, HUNGARIAN_CHICKENPOX, INDIAN_STOCK_MARKET]:
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

    dataset[DATA] = pd.DataFrame(
        StandardScaler().fit_transform(dataset[DATA]), 
        columns=dataset[DATA].columns, index=dataset[DATA].index
    )

    return dataset
            
def get_all_datasets():
    """
    Fetch and preprocess all datasets.

    :return dict: Dictionary containing all preprocessed datasets.
    """
    return {
        STOCK: preprocess(get_data(STOCK_DATASET_ID), STOCK),
        RMFTSA_LADATA: preprocess(get_data(RMFTSA_LADATA_DATASET_ID), RMFTSA_LADATA),
        CONCRETE_DATA: preprocess(get_data(CONCRETE_DATA_DATASET_ID), CONCRETE_DATA),
        DEBUTANIZER: preprocess(get_data(DEBUTANIZER_DATASET_ID), DEBUTANIZER),
        TREASURY: preprocess(get_data(TREASURY_DATASET_ID), TREASURY),
        WEATHER_IZMIR: preprocess(get_data(WEATHER_IZMIR_DATASET_ID), WEATHER_IZMIR),
        HUNGARIAN_CHICKENPOX: preprocess(get_data(HUNGARIAN_CHICKENPOX_DATASET_ID), HUNGARIAN_CHICKENPOX),
        CNN_STOCK_PRED_DJI: preprocess(get_data(CNN_STOCK_PRED_DJI_DATASET_ID), CNN_STOCK_PRED_DJI),
        DIABETES: preprocess(get_data(DIABETES_DATASET_ID), DIABETES),
        INDIAN_STOCK_MARKET: preprocess(get_data(INDIAN_STOCK_MARKET_DATASET_ID), INDIAN_STOCK_MARKET),
        GENDER_RECOGNITION: preprocess(get_data(GENDER_RECOGNITION_DATASET_ID), GENDER_RECOGNITION),
        BOSTON_WEATHER: preprocess(get_data(BOSTON_WEATHER_DATASET_ID), BOSTON_WEATHER),
        RED_WINE_QUALITY: preprocess(get_data(RED_WINE_QUALITY_DATASET_ID), RED_WINE_QUALITY),
        WHITE_WINE_QUALITY: preprocess(get_data(WHITE_WINE_QUALITY_DATASET_ID), WHITE_WINE_QUALITY),
        AIR_QUALITY: preprocess(get_data(AIR_QUALITY_DATASET_ID), AIR_QUALITY),
        FOOTBALL_PLAYER_POSITION: preprocess(get_data(FOOTBALL_PLAYER_POSITION_DATASET_ID), FOOTBALL_PLAYER_POSITION),
        TORUS: preprocess(torus(), TORUS),
        ON_AND_ON: preprocess(on_and_on(), ON_AND_ON)
    }