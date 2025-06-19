import numpy as np
import pandas as pd
import tadasets
import librosa

from src.constants import *

def torus():
    """
    Generate a torus dataset.
    """
    data = tadasets.torus(n=TORUS_SAMPLES, ambient=TORUS_AMBIENT, seed=TORUS_SEED)
    return {'data': pd.DataFrame(data, columns=[f"X{i}" for i in range(data.shape[1])])}

def on_and_on():
    """
    Generate a dataset from the 'On and On' audio file by computing chroma features
    with sliding windows. The resulting sequence of chroma vectors forms a point cloud
    representing the harmonic structure over time.
    """
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

    window_size = int(WINDOW_DURATION * sr / HOP_LENGTH)
    step_size = int(STEP_DURATION * sr / HOP_LENGTH)
    windows = np.stack([chroma[:, start:(start + window_size)] 
                        for start in range(0, chroma.shape[1] - window_size + 1, step_size)])

    return {'data': pd.DataFrame(windows.mean(axis=2), columns=CHROMA_LABELS)}