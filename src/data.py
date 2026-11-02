import numpy as np
import pandas as pd
import tadasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from src.constants import *
from scipy.spatial.distance import pdist
from src.config import MANIFOLD_PARAMS, SYNTHETIC_SWEEP
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

def _tanh_embed(data, ambient, seed):
    """Random 2-layer tanh network from the standardised sample to `ambient` dimensions (hidden width = ambient)."""
    rng = np.random.default_rng([seed, 4])
    z = (data - data.mean(axis=0)) / data.std(axis=0)
    hidden = np.tanh(z @ rng.normal(0, 1 / np.sqrt(data.shape[1]), (data.shape[1], ambient)) + rng.normal(0, 1, ambient))
    return hidden @ rng.normal(0, 1 / np.sqrt(ambient), (ambient, ambient))

def synthetic(manifold, m_noise=0, sigma=0.0, embedding='linear', seed=MANIFOLD_PARAMS['seed']):
    """
    Synthetic manifold with controlled feature redundancy and ambient noise, seeded throughout.
    The 3-d sample is embedded in ambient - m_noise columns (linear: seeded random rotation as for the study
    manifolds, so every column is a linear mixture of the 3 intrinsic coordinates; tanh: a random 2-layer network),
    m_noise columns of i.i.d. N(0, 1) noise are appended, and N(0, (sigma * diameter)^2) noise is added to every
    column, with the diameter of the 3-d sample. synthetic(m) with the defaults equals the study manifold m.

    :param str manifold: TORUS, SWISS_ROLL or SPHERE.
    :param int m_noise: Number of pure-noise columns.
    :param float sigma: Ambient noise level, relative to the manifold diameter.
    :param str embedding: 'linear' or 'tanh'.
    :param int seed: Seed of the sample, the embedding and the noise (separate streams).

    :return dict: The dataset (DATA as a DataFrame), not yet standardised.
    """
    n, ambient = MANIFOLD_PARAMS['samples'], MANIFOLD_PARAMS['ambient']
    sample = MANIFOLD_SAMPLERS[manifold](n=n, seed=seed)
    data = (embed if embedding == 'linear' else _tanh_embed)(sample, ambient - m_noise, seed)
    data = np.hstack([data, np.random.default_rng([seed, 2]).standard_normal((n, m_noise))])
    if sigma:
        data = data + sigma * pdist(sample).max() * np.random.default_rng([seed, 3]).standard_normal(data.shape)
    return {DATA: pd.DataFrame(data, columns=[f'X{i}' for i in range(data.shape[1])])}

def synthetic_variants():
    """
    Keys and generator arguments of the controlled synthetic sweep (config: synthetic_sweep): for every manifold,
    the redundancy sweep over pure-noise columns, the ambient-noise sweep and the nonlinear embedding.
    """
    variants = {}
    for manifold in SYNTHETIC_SWEEP['manifolds']:
        for m in SYNTHETIC_SWEEP['noise_columns']:
            variants[f'{manifold}_noise{m}'] = dict(manifold=manifold, m_noise=m)
        for sigma in SYNTHETIC_SWEEP['ambient_noise']:
            variants[f'{manifold}_sigma{sigma}'] = dict(manifold=manifold, sigma=sigma)
        if SYNTHETIC_SWEEP['nonlinear']:
            variants[f'{manifold}_tanh'] = dict(manifold=manifold, embedding='tanh')
    return variants

def get_synthetic_datasets():
    """
    Generate and preprocess every variant of the synthetic sweep.

    :return dict: Dictionary containing the preprocessed variants.
    """
    return {key: preprocess(synthetic(**args), key) for key, args in synthetic_variants().items()}

def get_dataset_set(name):
    """
    Generated or restricted dataset sets that experiments can name in the config (`datasets:`).

    :param str name: 'synthetic_sweep' or 'manifolds'.

    :return dict: Dictionary containing the preprocessed datasets.
    """
    if name == 'synthetic_sweep':
        return get_synthetic_datasets()
    if name == 'manifolds':
        return {DATASETS[id]: preprocess(get_data(id), DATASETS[id]) for id in (-1, -2, -3)}
    raise ValueError(f'Unknown dataset set: {name}')

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