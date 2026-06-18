import numpy as np
from scipy.spatial.distance import pdist

def diameter(points):
    return pdist(np.asarray(points)).max()

def normalize_by_diameter(X, points_x):
    X0 = np.asarray(X[0])
    X0_normalized = X0.copy()
    X0_normalized[:, 0:2] = X0[:, 0:2] / diameter(points_x)
    X[0] = X0_normalized
    return X