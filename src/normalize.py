import numpy as np

def _diameter(points):
    pts = np.asarray(points)
    dist_matrix = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return dist_matrix.max()

def normalize_by_diameter(X, points_x):
    X0 = np.asarray(X[0])
    X0_normalized = X0.copy()
    X0_normalized[:, 0:2] = X0[:, 0:2] / _diameter(points_x)
    X[0] = X0_normalized
    return X
