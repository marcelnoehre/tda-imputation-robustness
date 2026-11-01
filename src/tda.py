import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceImage
from sklearn.metrics.pairwise import pairwise_distances
from src.utils import as_batch, transform_pd
from src.constants import *
from src.config import N_JOBS, DIMENSIONS, MEL_PARAMS, WEIGHTED_RIPS_PARAMS, DTM_PARAMS, KD_PARAMS, LANDSCAPE_PARAMS, IMAGE_PARAMS

def _max_edge_length(distances):
    n, size = len(distances), MEL_PARAMS['subsample']
    if n > size:
        idx = np.random.default_rng(MEL_PARAMS['subsample_seed']).choice(n, size, replace=False)
        distances = distances[np.ix_(idx, idx)]
    tau_0 = float(minimum_spanning_tree(distances).max())
    return MEL_PARAMS['mst_scale_factor'] * tau_0

def distance_to_measure(distances, k):
    """
    Empirical distance to a measure (Chazal et al.) at every point, as in giotto-tda's 'DTM' weights:
    sqrt(sum of the squared distances to the k nearest neighbours / (k + 1)), the point itself being the (k+1)-th.
    """
    knn = np.partition(distances + np.diag(np.full(len(distances), np.inf)), k - 1, axis=1)[:, :k]
    return np.sqrt((knn**2).sum(axis=1) / (k + 1))

def kernel_distance_to_measure(distances, h):
    """
    Kernel distance to the empirical measure P (Phillips et al.) with a Gaussian kernel k(x, y) = exp(-|x - y|^2 / 2h^2),
    d_K(P, x_i) = sqrt(kappa(P, P) + k(x_i, x_i) - 2 kappa(P, x_i)) with kappa(P, x) = (1/n) sum_j k(x_j, x) and k(x, x) = 1.
    Scaled by h so that it is 1-Lipschitz in the Euclidean distance like the DTM: h -> infinity gives the distance to
    the centroid, h -> 0 a constant (Vietoris-Rips shifted by h).
    """
    K = np.exp(-distances**2 / (2 * h**2))
    kappa = K.mean(axis=1)
    return h * np.sqrt(np.maximum(K.mean() + 1 - 2 * kappa, 0))

def kernel_bandwidth(distances, d, tda):
    """Bandwidth of a kernel-distance filtration: a multiple of the median pairwise distance, or Silverman's rule."""
    factor = KD_PARAMS['bandwidths'][tda]
    if factor == 'silverman':
        n = len(distances)
        return (n * (d + 2) / 4) ** (-1 / (d + 4))
    return factor * float(np.median(distances[np.triu_indices(len(distances), 1)]))

def filtration_params(data, tda):
    """
    Parameters of a filtration that are fixed from the original cloud and shared by all its perturbed versions:
    the bandwidth of a kernel-distance filtration (empty for VR and DTM).
    """
    if tda in KD_PARAMS['bandwidths']:
        distances = pairwise_distances(data)
        return {'bandwidth': kernel_bandwidth(distances, np.shape(data)[1], tda)}
    return {}

def vertex_weights(data, distances, tda, bandwidth=None):
    """Vertex function f of the weighted Rips filtration (Anai et al.); the filtrations differ only here."""
    n, d = np.shape(data)
    if tda == DTMC:
        return distance_to_measure(distances, min(math.ceil(DTM_PARAMS['m'] * n), n - 1))
    if tda in KD_PARAMS['bandwidths']:
        return kernel_distance_to_measure(distances, kernel_bandwidth(distances, d, tda) if bandwidth is None else bandwidth)
    raise ValueError(f'{tda} is not a weighted Rips filtration')

def max_edge_length_for(data, tda):
    """
    Compute the filtration cutoff of the original cloud for a given TDA method: MST_SCALE_FACTOR times the longest
    edge of the Euclidean minimum spanning tree (on a random subsample). For the weighted filtrations it is raised
    to the largest vertex value, so that every vertex enters the filtration.
    """
    distances = pairwise_distances(data)
    cutoff = _max_edge_length(distances)
    if tda == VR:
        return cutoff
    # giotto-tda doubles the vertex values (diameter convention, as for VR edges)
    weights = 2 * vertex_weights(data, distances, tda)
    return max(cutoff, float(np.nextafter(weights.max(), np.inf)))

def vietoris_rips_complex(data, max_edge_length):
    return VietorisRipsPersistence(
        homology_dimensions=DIMENSIONS,
        max_edge_length=max_edge_length,
        collapse_edges=True,
        n_jobs=N_JOBS
    ).fit_transform(as_batch(data))

class _FixedWeights:
    """Picklable callable handing precomputed vertex weights to giotto-tda."""
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, _):
        return self.weights

def weighted_rips_complex(data, tda, max_edge_length, bandwidth=None):
    """
    Weighted Rips filtration (Anai et al.) on Euclidean distances with vertex function f = vertex_weights(., tda):
    a vertex enters at 2 f(x), an edge at the time its two balls of radius (t^p - f^p)^(1/p) meet (doubled, p = 2).
    DTM and kernel distance share this builder and differ only in f.
    """
    distances = pairwise_distances(data)
    return WeightedRipsPersistence(
        metric='precomputed',
        homology_dimensions=DIMENSIONS,
        max_edge_weight=max_edge_length,
        collapse_edges=True,
        weights=_FixedWeights(2 * vertex_weights(data, distances, tda, bandwidth)),
        weight_params={'p': WEIGHTED_RIPS_PARAMS['p']},
        n_jobs=N_JOBS
    ).fit_transform(distances[np.newaxis])

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
    DTMC: {FUNCTION: lambda data, mel: weighted_rips_complex(data, DTMC, mel)},
    **{kd: {FUNCTION: lambda data, mel, bandwidth, kd=kd: weighted_rips_complex(data, kd, mel, bandwidth)} for kd in KD_PARAMS['bandwidths']},
    PD: {FUNCTION: lambda data: transform_pd(data)},
    DIAGRAM: {FUNCTION: lambda data: np.asarray(data)}
}