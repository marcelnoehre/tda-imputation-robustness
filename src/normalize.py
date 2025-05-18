import numpy as np

def _compute_diameter(points):
    pts = np.asarray(points)
    dist_matrix = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return dist_matrix.max()

def normalize_by_diameter(X, Y, points_x, points_y):
    X_normalized = X / _compute_diameter(points_x)
    Y_normalized = Y / _compute_diameter(points_y)
    return X_normalized, Y_normalized