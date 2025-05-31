from persim import wasserstein, bottleneck
import numpy as np

def compute_wasserstein_distance(X, Y):
    return wasserstein(X, Y)

def compute_bottleneck_distance(X, Y):
    return bottleneck(X, Y)

def landscape_l2_distance(landscape1: np.ndarray, landscape2: np.ndarray, delta_t=1.0) -> float:
    if landscape1.shape != landscape2.shape:
        raise ValueError("Landscapes must have the same shape")
    return np.sqrt(np.sum((landscape1 - landscape2)**2) * delta_t)

def persistence_image_l2_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        raise ValueError("Persistence images must have the same shape")
    return np.linalg.norm(img1 / np.linalg.norm(img1) - img2 / np.linalg.norm(img2))

# TODO: Check Normalization
def _handle_infinite_deaths(diag):
    finite_deaths = diag[np.isfinite(diag[:,1]), 1]
    max_finite = np.max(finite_deaths) if finite_deaths.size > 0 else 1.0
    replacement = max_finite + 1.0
    diag_fixed = diag.copy()
    diag_fixed[~np.isfinite(diag_fixed[:,1]), 1] = replacement
    return diag_fixed

def max_persistence(diag):
    if diag.size == 0:
        return 0.0
    return np.max(diag[:,1] - diag[:,0])

def compute_normalized_wasserstein_distance(X, Y, eps=1e-6):
    X, Y = _handle_infinite_deaths(X), _handle_infinite_deaths(Y)
    denom = (max_persistence(X) + max_persistence(Y)) / 2
    denom = max(denom, eps)
    return wasserstein(X, Y) / denom

def compute_normalized_bottleneck_distance(X, Y, eps=1e-6):
    X, Y = _handle_infinite_deaths(X), _handle_infinite_deaths(Y)
    denom = (max_persistence(X) + max_persistence(Y)) / 2
    denom = max(denom, eps)
    return bottleneck(X, Y) / denom