from persim import wasserstein, bottleneck
import numpy as np

def compute_wasserstein_distance(X, Y):
    return wasserstein(X, Y)

def compute_bottleneck_distance(X, Y):
    return bottleneck(X, Y)

# TODO: Check Normalization
def _handle_infinite_deaths(diag):
    finite_deaths = diag[np.isfinite(diag[:,1]), 1]
    max_finite = np.max(finite_deaths) if finite_deaths.size > 0 else 1.0
    replacement = max_finite + 1.0
    diag_fixed = diag.copy()
    diag_fixed[~np.isfinite(diag_fixed[:,1]), 1] = replacement
    return diag_fixed

def _dist_to_diagonal(diag):
    return np.sum((diag[:,1] - diag[:,0]) / 2)

def compute_normalized_wasserstein_distance(X, Y):
    X, Y = _handle_infinite_deaths(X), _handle_infinite_deaths(Y)
    return wasserstein(X, Y) / ((_dist_to_diagonal(X) + _dist_to_diagonal(Y)) / 2)

def compute_normalized_bottleneck_distance(X, Y):
    X, Y = _handle_infinite_deaths(X), _handle_infinite_deaths(Y)
    return bottleneck(X, Y) / ((_dist_to_diagonal(_handle_infinite_deaths(X)) + _dist_to_diagonal(_handle_infinite_deaths(Y))) / 2)