import numpy as np
from persim import wasserstein, bottleneck
from src.tda import persistence_landscape, persistence_image
from src.utils import transform_pd
from src.constants import *

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

def compute_wasserstein_distance(X, Y):
    return wasserstein(X, Y)

def compute_bottleneck_distance(X, Y):
    return bottleneck(X, Y)

def landscape_l2_distance(X: np.ndarray, Y: np.ndarray, pd):
    if X.shape != Y.shape:
        raise ValueError(f'Landscapes must have same shape, got {X.shape} vs {Y.shape}')
    diff = X - Y
    bd = np.array(pd[0])[:, :2]
    t_min = np.min(bd[:, 0])
    t_max = np.max(bd[:, 1])
    delta_t = (t_max - t_min) / (X.shape[0] - 1)
    return np.sqrt(np.sum(diff**2) * delta_t)

def persistence_image_l2_distance(X: np.ndarray, Y: np.ndarray):
    if X.shape != Y.shape:
        raise ValueError(f'Persistence images must have same shape, got {X.shape} vs {Y.shape}')
    return np.linalg.norm(X - Y)

METRICS = {
    RMSE: {FUNCTION: lambda X, Y: compute_rmse(X, Y)},
    MAE: {FUNCTION: lambda X, Y: compute_mae(X, Y)},
    WS: {FUNCTION: lambda X, Y, dim: compute_wasserstein_distance(X[PD][dim], transform_pd(Y)[dim])},
    BN: {FUNCTION: lambda X, Y, dim: compute_bottleneck_distance(X[PD][dim], transform_pd(Y)[dim])},
    L2PL: {FUNCTION: lambda X, Y, dim: landscape_l2_distance(X[PL][dim], persistence_landscape(Y)[dim], X[PD])},
    L2PI: {FUNCTION: lambda X, Y, dim: persistence_image_l2_distance(X[PI][dim], persistence_image(Y)[dim])}
}
