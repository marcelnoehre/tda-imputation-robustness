import numpy as np

def numeric_target_mapping(target):
    return np.array([{cls: id for id, cls in enumerate(target)}[cls] for cls in target])

def as_batch(data):
    return np.array(data)[None, :, :]
