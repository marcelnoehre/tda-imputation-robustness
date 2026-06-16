import math
import numpy as np
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceImage
from sklearn.metrics.pairwise import pairwise_distances
from src.utils import as_batch, transform_pd
from src.constants import *

def _max_edge_length(distances):
    n = len(distances)
    if n > 500:
        idx = np.random.default_rng(42).choice(n, 500, replace=False)
        distances = distances[np.ix_(idx, idx)]
    upper = distances[np.triu_indices_from(distances, k=1)]
    return float(np.percentile(upper, EDGE_LENGTH_PERCENTILE))

def max_edge_length_for(data, tda):
    """Compute the filtration cutoff from the original data for a given TDA method."""
    if tda == KD:
        n, d = data.shape
        h = (n * (d + 2) / 4) ** (-1 / (d + 4))
        K = np.exp(-pairwise_distances(data)**2 / (2 * h**2))
        return _max_edge_length(np.sqrt(2 - 2 * K))
    return _max_edge_length(pairwise_distances(data))

def vietoris_rips_complex(data, max_edge_length):
    return VietorisRipsPersistence(
        homology_dimensions=DIMENSIONS,
        max_edge_length=max_edge_length,
        collapse_edges=True,
        n_jobs=N_JOBS
    ).fit_transform(as_batch(data))

def distance_to_a_measure(data, chazal, max_edge_length):
    n = len(data)
    if chazal:
        m = 0.05
        k = math.ceil(m * n)
    else:
        k = min(int(np.sqrt(n)), n // 2)
    return WeightedRipsPersistence(
        homology_dimensions=DIMENSIONS,
        max_edge_weight=max_edge_length,
        collapse_edges=True,
        weights='DTM',
        weight_params={
            'n_neighbors': k,
            'p': EUCLIDEAN
        },
        n_jobs=N_JOBS
    ).fit_transform(as_batch(data))

def kernel_distance(data, max_edge_length):
    n, d = data.shape
    h = (n * (d + 2) / 4) ** (-1 / (d + 4))
    distances = pairwise_distances(data)
    K = np.exp(-distances**2 / (2 * h**2))
    D = np.sqrt(2 - 2 * K)
    return VietorisRipsPersistence(
        metric='precomputed',
        homology_dimensions=DIMENSIONS,
        max_edge_length=max_edge_length,
        collapse_edges=True,
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
    VR: {FUNCTION: lambda data, mel: vietoris_rips_complex(data, mel)},
    DTMS: {FUNCTION: lambda data, mel: distance_to_a_measure(data, False, mel)},
    DTMC: {FUNCTION: lambda data, mel: distance_to_a_measure(data, True, mel)},
    KD: {FUNCTION: lambda data, mel: kernel_distance(data, mel)},
    PD: {FUNCTION: lambda data: transform_pd(data)},
    PL: {FUNCTION: lambda data: persistence_landscape(data)},
    PI: {FUNCTION: lambda data: persistence_image(data)}
}