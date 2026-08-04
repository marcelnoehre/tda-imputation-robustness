import os
import functools
import threading
import numpy as np
import pandas as pd
import h5py
import joblib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_ROOT, '.cache')

_SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)

_MISS = object()


def _entry_dir(func, args, kwargs):
    key = joblib.hash((args, kwargs), coerce_mmap=True)
    return os.path.join(_CACHE_DIR, func.__module__, func.__qualname__, key)


def _atomic_write(path, write_to):
    tmp_path = f'{path}.tmp-{os.getpid()}-{threading.get_ident()}'
    write_to(tmp_path)
    os.replace(tmp_path, path)


def _save_ndarray(path, arr):
    def write(tmp_path):
        with open(tmp_path, 'wb') as f:
            np.save(f, arr, allow_pickle=False)
    _atomic_write(path, write)


def _save_array_sequence(path, arrays):
    def write(tmp_path):
        with open(tmp_path, 'wb') as f:
            np.savez(f, *[np.asarray(a, dtype=np.float64) for a in arrays])
    _atomic_write(path, write)


def _save_dataframe(path, df):
    def write(tmp_path):
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('values', data=df.to_numpy(dtype=np.float64))
            f.create_dataset('columns', data=list(df.columns.astype(str)), dtype=h5py.string_dtype())
            if not df.index.equals(pd.RangeIndex(len(df))):
                f.create_dataset('index', data=df.index.to_numpy())
    _atomic_write(path, write)


def _store(entry_dir, value):
    os.makedirs(entry_dir, exist_ok=True)
    if isinstance(value, pd.DataFrame):
        _save_dataframe(os.path.join(entry_dir, 'output.h5'), value)
    elif isinstance(value, np.ndarray):
        _save_ndarray(os.path.join(entry_dir, 'output.npy'), value)
    elif isinstance(value, _SCALAR_TYPES):
        _save_ndarray(os.path.join(entry_dir, 'output.npy'), np.asarray(value))
    elif isinstance(value, (list, tuple)):
        _save_array_sequence(os.path.join(entry_dir, 'output.npz'), value)
    else:
        raise TypeError(
            f'src.cache cannot persist a {type(value)!r} result without pickling; '
            'add a serializer for this type in src/cache.py.'
        )


def _load(entry_dir):
    npy_path = os.path.join(entry_dir, 'output.npy')
    if os.path.exists(npy_path):
        arr = np.load(npy_path, allow_pickle=False)
        return float(arr) if arr.ndim == 0 else arr

    npz_path = os.path.join(entry_dir, 'output.npz')
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            names = sorted(data.files, key=lambda n: int(n.split('_')[1]))
            return [data[n] for n in names]

    h5_path = os.path.join(entry_dir, 'output.h5')
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            values = f['values'][:]
            columns = [c.decode('utf-8') if isinstance(c, bytes) else c for c in f['columns'][:]]
            index = f['index'][:] if 'index' in f else None
        return pd.DataFrame(values, columns=columns, index=index)

    return _MISS


def _cached(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        entry_dir = _entry_dir(func, args, kwargs)
        value = _load(entry_dir)
        if value is _MISS:
            value = func(*args, **kwargs)
            _store(entry_dir, value)
        return value
    return wrapper


class _Cache:
    cache = staticmethod(_cached)


memory = _Cache()
