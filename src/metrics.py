import numpy as np
import gudhi
from persim import wasserstein
from src.tda import persistence_landscape, persistence_image
from src.utils import transform_pd
from src.constants import *

# RMSE and MAE run over all entries (observed ones contribute zero), i.e. the total perturbation
# of the cloud, which is what the topological distances compare; masked-entry RMSE = RMSE / sqrt(MR).
def compute_rmse(X: np.ndarray, Y: np.ndarray):
    if X.shape != Y.shape:
        raise ValueError(f'Arrays must have same shape, got {X.shape} vs {Y.shape}')
    diff = X - Y
    return np.sqrt(np.mean(diff**2))

def compute_mae(X: np.ndarray, Y: np.ndarray):
    if X.shape != Y.shape:
        raise ValueError(f'Arrays must have same shape, got {X.shape} vs {Y.shape}')
    diff = np.abs(X - Y)
    return np.mean(diff)

def compute_max_displacement(X: np.ndarray, Y: np.ndarray):
    # e_inf = max_i |x_i - y_i|_2, the quantity in the Rips stability bound d_B <= 2 e_inf
    if X.shape != Y.shape:
        raise ValueError(f'Arrays must have same shape, got {X.shape} vs {Y.shape}')
    return np.linalg.norm(X - Y, axis=1).max()

def compute_wasserstein_distance(X, Y):
    return wasserstein(X, Y)

def compute_bottleneck_distance(X, Y):
    return gudhi.bottleneck_distance(X, Y)

def landscape_l2_distance(X: np.ndarray, Y: np.ndarray, sampling: np.ndarray):
    # X and Y are sampled on the same grid `sampling` (see tda.persistence_landscape)
    if X.shape != Y.shape or X.shape[-1] != len(sampling):
        raise ValueError(f'Landscapes must share one grid, got {X.shape} vs {Y.shape} on {len(sampling)} samples')
    delta_t = sampling[1] - sampling[0]
    return np.sqrt(np.sum((X - Y)**2) * delta_t)

def persistence_image_l2_distance(X: np.ndarray, Y: np.ndarray, sampling: np.ndarray):
    # X and Y are sampled on the same grid `sampling` (see tda.persistence_image)
    if X.shape != Y.shape or X.shape != (len(sampling[0]), len(sampling[1])):
        raise ValueError(f'Persistence images must share one grid, got {X.shape} vs {Y.shape}')
    return np.linalg.norm(X - Y)

def _landscape_l2(X, Y, dim):
    pl_X, pl_Y, samplings = persistence_landscape(X[DIAGRAM], Y)
    return landscape_l2_distance(pl_X[dim], pl_Y[dim], samplings[dim])

def _image_l2(X, Y, dim):
    pi_X, pi_Y, samplings = persistence_image(X[DIAGRAM], Y)
    return persistence_image_l2_distance(pi_X[dim], pi_Y[dim], samplings[dim])

METRICS = {
    RMSE: {FUNCTION: lambda X, Y: compute_rmse(X, Y)},
    MAE: {FUNCTION: lambda X, Y: compute_mae(X, Y)},
    MAXDISP: {FUNCTION: lambda X, Y: compute_max_displacement(X, Y)},
    WS: {FUNCTION: lambda X, Y, dim: compute_wasserstein_distance(X[PD][dim], transform_pd(Y)[dim])},
    BN: {FUNCTION: lambda X, Y, dim: compute_bottleneck_distance(X[PD][dim], transform_pd(Y)[dim])},
    L2PL: {FUNCTION: lambda X, Y, dim: _landscape_l2(X, Y, dim)},
    L2PI: {FUNCTION: lambda X, Y, dim: _image_l2(X, Y, dim)}
}
