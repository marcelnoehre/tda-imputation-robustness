import pandas as pd
import tadasets

from src.constants import *

def torus():
    data = tadasets.torus(n=TORUS_SAMPLES, ambient=TORUS_AMBIENT, seed=TORUS_SEED)
    return {'data': pd.DataFrame(data, columns=[f"X{i}" for i in range(data.shape[1])])}