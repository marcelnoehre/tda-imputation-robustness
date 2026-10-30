import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceImage
from sklearn.metrics.pairwise import pairwise_distances
from src.utils import as_batch, transform_pd
from src.constants import *
from src.config import N_JOBS, DIMENSIONS, MEL_PARAMS, DTM_PARAMS, LANDSCAPE_PARAMS, IMAGE_PARAMS

def _max_edge_length(distances):
    n, size = len(distances), MEL_PARAMS['subsample']
    if n > size:
        idx = np.random.default_rng(MEL_PARAMS['subsample_seed']).choice(n, size, replace=False)
        distances = distances[np.ix_(idx, idx)]
    tau_0 = float(minimum_spanning_tree(distances).max())
    return MEL_PARAMS['mst_scale_factor'] * tau_0

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

def distance_to_a_measure(data, max_edge_length):
    n = len(data)
    k = min(math.ceil(DTM_PARAMS['m'] * n), n - 1)
    distances = pairwise_distances(data)
    np.fill_diagonal(distances, np.inf)
    kth_nn_dist = float(np.sort(distances, axis=1)[:, k - 1].max())
    np.fill_diagonal(distances, 0.0)
    # Pass distances as precomputed so ripser uses the exact same values we computed kth_nn_dist from.
    # nextafter ensures the k-th neighbor is strictly inside the threshold even if ripser uses strict <.
    effective_mel = max(max_edge_length, np.nextafter(kth_nn_dist, np.inf))
    return WeightedRipsPersistence(
        metric='precomputed',
        homology_dimensions=DIMENSIONS,
        max_edge_weight=effective_mel,
        collapse_edges=True,
        weights='DTM',
        weight_params={
            'n_neighbors': k,
            'p': DTM_PARAMS['p']
        },
        n_jobs=N_JOBS
    ).fit_transform(distances[np.newaxis])

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

def _on_common_grid(vectorizer, pd_fit, pd):
    # The grid is fitted once on pd_fit (the original diagram) and both diagrams are transformed with it,
    # so the two vectors share one grid by construction.
    vectorizer.fit(pd_fit)
    return vectorizer.transform(pd_fit)[0], vectorizer.transform(pd)[0], vectorizer.samplings_

def persistence_landscape(pd_fit, pd):
    return _on_common_grid(PersistenceLandscape(**LANDSCAPE_PARAMS, n_jobs=N_JOBS), pd_fit, pd)

def persistence_image(pd_fit, pd):
    return _on_common_grid(PersistenceImage(**IMAGE_PARAMS, n_jobs=N_JOBS), pd_fit, pd)

TDA = {
    VR: {FUNCTION: lambda data, mel: vietoris_rips_complex(data, mel)},
    DTMC: {FUNCTION: lambda data, mel: distance_to_a_measure(data, mel)},
    KD: {FUNCTION: lambda data, mel: kernel_distance(data, mel)},
    PD: {FUNCTION: lambda data: transform_pd(data)},
    DIAGRAM: {FUNCTION: lambda data: np.asarray(data)}
}