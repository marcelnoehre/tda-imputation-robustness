import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import pdist, squareform
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceImage
from src.utils import as_batch

def vietoris_rips_complex(data, maxdim=2):
    return VietorisRipsPersistence(
        homology_dimensions=list(range(maxdim+1))
    ).fit_transform(as_batch(data))

def distance_to_a_measure(data, r=2, p=2, maxdim=2):
    n = len(data)
    k = int(max(10, min(np.sqrt(n), n // 2)))
    return WeightedRipsPersistence(
        homology_dimensions=list(range(maxdim+1)),
        weights="DTM",
        weight_params={"n_neighbors": k, "r": r, "p": p}
    ).fit_transform(as_batch(data))

def kernel_distance(data, maxdim=2):
    X = np.asarray(data)
    # cross-validated KDE bandwidth
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": np.logspace(-1, 1, 20)},
    )
    grid.fit(X)
    h = grid.best_params_["bandwidth"]

    # Gram matrix K_{i,j}
    pairwise_sq = squareform(pdist(X, metric="sqeuclidean"))
    K = np.exp(-pairwise_sq / (2 * h**2))

    n = len(X)
    # (1/n^2) * sum_{i=1}^{n} sum_{j=1}^{n} K(X_i, X_j)
    const = K.sum() / n**2
    # K(x, x) is always 1 for Gaussian kernel (since ||x - x|| = 0)

    # (2/n) * sum_{i=1}^{n} K(x, X_i), for each x = X_i
    row_sums = K.sum(axis=1)
    # d_K^2(x_i) for all x_i
    d2 = 1 + const - (2 / n) * row_sums
    # weights = sqrt of d_K^2(x_i) -> no negative weights
    weights = np.sqrt(np.maximum(d2, 0))

    return WeightedRipsPersistence(
        homology_dimensions=list(range(maxdim + 1)),
        metric="euclidean",
        weights=lambda pc, w=weights: w # skip point cloud, use weights
    ).fit_transform(as_batch(data))

def persistence_landscape(pd):
    return PersistenceLandscape().fit_transform(pd)[0]

def persistence_image(pd):
    return PersistenceImage().fit_transform(pd)
