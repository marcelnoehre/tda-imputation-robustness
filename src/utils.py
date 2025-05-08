import numpy as np

def map_target_to_np_array(dataset, target):
    class_names = np.unique(dataset[target])
    class_to_number = {class_name: idx for idx, class_name in enumerate(class_names)}
    return np.array([class_to_number[class_name] for class_name in dataset[target]])