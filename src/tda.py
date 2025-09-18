import math
import numpy as np
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceImage
from sklearn.metrics.pairwise import pairwise_distances
from src.utils import as_batch, transform_pd
from src.constants import *

def vietoris_rips_complex(data):
    return VietorisRipsPersistence(
        homology_dimensions=DIMENSIONS,
        n_jobs=N_JOBS
    ).fit_transform(as_batch(data))

def distance_to_a_measure(data, chazal):
    n = len(data)
    if chazal:
        m = 0.05
        k = math.ceil(m * n)
    else:
        k = min(int(np.sqrt(n)), n // 2)
    return WeightedRipsPersistence(
        homology_dimensions=DIMENSIONS,
        weights='DTM',
        weight_params={
            'n_neighbors': k,
            'p': EUCLIDEAN
        },
        n_jobs=N_JOBS
    ).fit_transform(as_batch(data))

def kernel_distance(data):
    n, d = data.shape
    h = (n * (d + 2) / 4) ** (-1 / (d + 4))
    distances = pairwise_distances(data)
    K = np.exp(-distances**2 / (2 * h**2))
    D = np.sqrt(2 - 2 * K)
    return VietorisRipsPersistence(
        metric='precomputed',
        homology_dimensions=DIMENSIONS
    ).fit_transform([D])

def persistence_landscape(pd):
    return PersistenceLandscape(
        n_jobs=N_JOBS,
    ).fit_transform(pd)[0]

def persistence_image(pd):
    return PersistenceImage(
        n_jobs=N_JOBS,
    ).fit_transform(pd)[0]

TDA = {
    VR: {FUNCTION: lambda data: vietoris_rips_complex(data)},
    DTMS: {FUNCTION: lambda data: distance_to_a_measure(data, False)},
    DTMC: {FUNCTION: lambda data: distance_to_a_measure(data, True)},
    KD: {FUNCTION: lambda data: kernel_distance(data)},
    PD: {FUNCTION: lambda data: transform_pd(data)},
    PL: {FUNCTION: lambda data: persistence_landscape(data)},
    PI: {FUNCTION: lambda data: persistence_image(data)}
}