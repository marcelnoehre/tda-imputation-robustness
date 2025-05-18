from persim import wasserstein, bottleneck

def compute_wasserstein_distance(X, Y):
    return wasserstein(X, Y)

def compute_bottleneck_distance(X, Y):
    return bottleneck(X, Y)
