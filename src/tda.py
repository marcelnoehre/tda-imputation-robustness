import numpy as np
from sklearn.neighbors import KernelDensity
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceImage
from src.utils import as_pointcloud

def vietoris_rips_complex(maxdim=2):
    return VietorisRipsPersistence(homology_dimensions=list(range(maxdim+1)))

def distance_to_a_measure(maxdim=2):
    return WeightedRipsPersistence(homology_dimensions=list(range(maxdim+1)))

def kernel_density_estimation(data, maxdim=2):
    X = as_pointcloud(data)
    n, d = X.shape
    sigma = np.mean(np.std(X, axis=0, ddof=1))
    bandwidth = 1.06 * sigma * (n ** (-1.0 / (d + 4)))
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X)

    log_densities = kde.score_samples(X)
    log_max = log_densities.max()

    def density_weights(point_cloud):
        return log_max - kde.score_samples(point_cloud)

    return WeightedRipsPersistence(
        homology_dimensions=list(range(maxdim + 1)),
        weights=density_weights
    )

def persistence_landscape():
    return PersistenceLandscape()

def persistence_image():
    return PersistenceImage()