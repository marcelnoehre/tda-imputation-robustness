from sklearn.metrics import mean_squared_error

def mse(original, restored, mask):
    return mean_squared_error(original[mask], restored[mask])