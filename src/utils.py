import numpy as np

def numeric_target_mapping(target):
    """
    Map target class labels to numeric values.

    :param list target: List of class labels.

    :return np.ndarray: Numeric mapping of class labels.
    """
    return np.array([{cls: id for id, cls in enumerate(target)}[cls] for cls in target])

def as_batch(data):
    return np.array(data)[None, :, :]

def transform_pd(data):
    return [[[b, d] for b, d, dim in data[0] if dim == i] for i in range(3)]