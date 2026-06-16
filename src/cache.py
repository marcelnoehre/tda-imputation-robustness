import os
import joblib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
memory = joblib.Memory(os.path.join(_ROOT, '.cache'), verbose=0)
