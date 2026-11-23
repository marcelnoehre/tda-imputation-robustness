import os
import sys

import numpy as np
import pandas as pd

import src.cache as cache
from src.constants import *


def _dataset(n=60, d=6):
    rng = np.random.default_rng(0)
    return {DATA: pd.DataFrame(rng.normal(size=(n, d)), columns=[f'X{i}' for i in range(d)]), TARGET: np.zeros(n)}


def _ph_entries(cache_dir):
    root = os.path.join(cache_dir, 'src.compute', 'apply_persistent_homology')
    return set(os.listdir(root)) if os.path.isdir(root) else set()


def _run(monkeypatch, tmp_path, name):
    for m in ('src.missingness', 'src.pipeline', 'src.robustness_pipeline'):
        __import__(m)
        monkeypatch.setattr(sys.modules[m], 'SEEDS', [1, 2], raising=False)
        monkeypatch.setattr(sys.modules[m], 'WORKERS', 1, raising=False)
    import src.robustness_pipeline as R
    monkeypatch.chdir(tmp_path)
    R.experiment_baselines(name, [MAR], [5, 10, 25], [VR], [WS, BN], [RANDOM_SUBSAMPLE_70, MATCHED_JITTER], {'toy': _dataset()})
    return pd.read_csv(tmp_path / 'results' / f'{name}_results.csv')


def test_rate_independent_clouds_get_one_diagram(tmp_path, monkeypatch):
    """RS 70% draws the same rows at every missing rate: one diagram per seed, shared by all rates, same results."""
    import src.pipeline as P
    calls = []
    unique = P._unique_ph_tasks
    monkeypatch.setattr(P, '_unique_ph_tasks', lambda tasks, fp: calls.append(len(tasks)) or unique(tasks, fp))
    monkeypatch.setenv('TDA_CACHE_DIR', str(tmp_path / 'a'))
    monkeypatch.setattr(cache, '_CACHE_DIR', str(tmp_path / 'a'))
    shared = _run(monkeypatch, tmp_path, 'dedup')
    # 2 seeds x 3 rates x (RS 70% + jitter) = 12 tasks; RS 70% needs one diagram per seed instead of three
    assert calls == [12]
    assert len(_ph_entries(tmp_path / 'a')) == 1 + 2 + 6   # original cloud, RS 70% per seed, jitter per seed and rate

    # without deduplication (fresh cache): every task computes its own diagram, and the results are the same
    monkeypatch.setattr(P, '_unique_ph_tasks', lambda tasks, fp: (tasks, {}))
    monkeypatch.setenv('TDA_CACHE_DIR', str(tmp_path / 'b'))
    monkeypatch.setattr(cache, '_CACHE_DIR', str(tmp_path / 'b'))
    separate = _run(monkeypatch, tmp_path, 'separate')
    pd.testing.assert_frame_equal(shared, separate)
    # RS 70% has the same distances at every rate
    rs = shared[shared['imputation_method'] == RANDOM_SUBSAMPLE_70]
    assert rs['missing_rate'].nunique() == 3
    assert (rs.groupby('dimension')['wasserstein_distance'].nunique() == 1).all()


def test_cached_only_skips_uncached_diagrams(tmp_path, monkeypatch):
    """--cached-only: settings whose diagram is not cached are left out of the results, the others are unchanged."""
    import src.pipeline as P
    import src.robustness_pipeline as R
    monkeypatch.setenv('TDA_CACHE_DIR', str(tmp_path / 'c'))
    monkeypatch.setattr(cache, '_CACHE_DIR', str(tmp_path / 'c'))
    for m in ('src.missingness', 'src.pipeline', 'src.robustness_pipeline'):
        monkeypatch.setattr(sys.modules[m], 'SEEDS', [1, 2], raising=False)
        monkeypatch.setattr(sys.modules[m], 'WORKERS', 1, raising=False)
    monkeypatch.chdir(tmp_path)
    run = lambda name, rates: (R.experiment_baselines(name, [MAR], rates, [VR], [WS, BN], [RANDOM_SUBSAMPLE_70, MATCHED_JITTER], {'toy': _dataset()}),
                               pd.read_csv(tmp_path / 'results' / f'{name}_distances.csv'))[1]
    full = run('full', [5, 10])
    entries = _ph_entries(tmp_path / 'c')

    monkeypatch.setattr(P, 'CACHED_ONLY', True)
    partial = run('partial', [5, 10, 25])
    assert _ph_entries(tmp_path / 'c') == entries   # nothing computed
    # jitter at 25% has no cached diagram and is skipped; RS 70% shares its cloud across rates, so it is cached
    at25 = partial[partial['missing_rate'] == 25]
    assert set(at25['imputation_method']) == {RANDOM_SUBSAMPLE_70}
    pd.testing.assert_frame_equal(partial[partial['missing_rate'] < 25].reset_index(drop=True), full)
