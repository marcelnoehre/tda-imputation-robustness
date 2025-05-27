import numpy as np
from sklearn.neighbors import KernelDensity
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence

def vietoris_rips_complex(data, maxdim=2):
    vr = VietorisRipsPersistence(homology_dimensions=list(range(maxdim+1)))
    return vr.fit_transform(np.array(data)[None, :, :])

def distance_to_a_measure(data, maxdim=2):
    dtm = WeightedRipsPersistence(homology_dimensions=list(range(maxdim+1)))
    return dtm.fit_transform(np.array(data)[None, :, :])

def kernel_density_estimation(data, maxdim=2):
    X = np.asarray(data)
    if X.ndim == 1:
        X = X[:, None]
    
    n, d = X.shape
    sigma = np.mean(np.std(X, axis=0, ddof=1))
    bandwidth = 1.06 * sigma * (n ** (-1.0 / (d + 4)))
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X)

    log_densities = kde.score_samples(X)
    log_max = log_densities.max()

    def density_weights(point_cloud):
        return log_max - kde.score_samples(point_cloud)

    wr = WeightedRipsPersistence(
        homology_dimensions=list(range(maxdim + 1)),
        weights=density_weights
    )
    return wr.fit_transform([X])