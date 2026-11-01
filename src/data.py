import numpy as np
import pandas as pd
import tadasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from src.constants import *
from src.config import MANIFOLD_PARAMS
from src.utils import numeric_target_mapping

def embed(data, ambient, seed):
    """
    Pad the data with zero columns to `ambient` dimensions and rotate it by a random orthogonal matrix.
    Same construction as tadasets.embed, whose rotation is drawn from the unseeded global generator.

    :param np.ndarray data: The sample, shape (n, d) with d < ambient.
    :param int ambient: The ambient dimension.
    :param int seed: Seed of the rotation (drawn from its own stream, independent of the sample).

    :return np.ndarray: The embedded data, shape (n, ambient).
    """
    n, d = data.shape
    base = np.zeros((n, ambient))
    base[:, :d] = data
    q, _ = np.linalg.qr(np.random.default_rng([seed, 1]).random((ambient, ambient)))
    return base @ q

def _embedded(sample, sample_seed):
    return embed(sample(n=MANIFOLD_PARAMS['samples'], seed=sample_seed), MANIFOLD_PARAMS['ambient'], MANIFOLD_PARAMS['seed'])

def _manifold(sample):
    data = _embedded(sample, MANIFOLD_PARAMS['seed'])
    return {DATA: pd.DataFrame(data, columns=[f'X{i}' for i in range(data.shape[1])])}

def torus():
    """
    Generate a torus dataset.
    """
    return _manifold(tadasets.torus)

def swiss_roll():
    """
    Generate a Swiss roll dataset.
    """
    return _manifold(tadasets.swiss_roll)

def sphere():
    """
    Generate a sphere dataset.
    """
    return _manifold(tadasets.sphere)

MANIFOLD_SAMPLERS = {TORUS: tadasets.torus, SWISS_ROLL: tadasets.swiss_roll, SPHERE: tadasets.sphere}

def fresh_sample(key, seed):
    """
    Draw a new sample of a synthetic manifold: same size, rotation and standardisation as the original,
    sample seed [manifold seed, seed]. Used as the sampling-noise floor.

    :param str key: The manifold key.
    :param int seed: The seed of the new sample.

    :return np.ndarray: The standardised sample.
    """
    sample = MANIFOLD_SAMPLERS[key]
    scaler = StandardScaler().fit(_embedded(sample, MANIFOLD_PARAMS['seed']))
    return scaler.transform(_embedded(sample, [MANIFOLD_PARAMS['seed'], seed]))

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
        if key in [HUNGARIAN_CHICKENPOX, INDIAN_STOCK_MARKET]:
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
        DIABETES: preprocess(get_data(DIABETES_DATASET_ID), DIABETES),
        INDIAN_STOCK_MARKET: preprocess(get_data(INDIAN_STOCK_MARKET_DATASET_ID), INDIAN_STOCK_MARKET),
        GENDER_RECOGNITION: preprocess(get_data(GENDER_RECOGNITION_DATASET_ID), GENDER_RECOGNITION),
        BOSTON_WEATHER: preprocess(get_data(BOSTON_WEATHER_DATASET_ID), BOSTON_WEATHER),
        RED_WINE_QUALITY: preprocess(get_data(RED_WINE_QUALITY_DATASET_ID), RED_WINE_QUALITY),
        WHITE_WINE_QUALITY: preprocess(get_data(WHITE_WINE_QUALITY_DATASET_ID), WHITE_WINE_QUALITY),
        AIR_QUALITY: preprocess(get_data(AIR_QUALITY_DATASET_ID), AIR_QUALITY),
        FOOTBALL_PLAYER_POSITION: preprocess(get_data(FOOTBALL_PLAYER_POSITION_DATASET_ID), FOOTBALL_PLAYER_POSITION),
        TORUS: preprocess(torus(), TORUS),
        SWISS_ROLL: preprocess(swiss_roll(), SWISS_ROLL),
        SPHERE: preprocess(sphere(), SPHERE)
    }