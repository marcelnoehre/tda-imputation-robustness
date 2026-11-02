import os
import sys

import numpy as np
import pandas as pd
import pytest

import src.cache as cache
from src.constants import *


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    # the environment variable reaches spawned worker processes, the attribute this process
    monkeypatch.setenv('TDA_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(cache, '_CACHE_DIR', str(tmp_path / 'cache'))
    return tmp_path / 'cache'


def _entries(cache_dir):
    return {root for root, _, files in os.walk(cache_dir) if any(f.startswith('output.') for f in files)}


def _dataset(n=60, d=6):
    rng = np.random.default_rng(0)
    return {DATA: pd.DataFrame(rng.normal(size=(n, d)), columns=[f'X{i}' for i in range(d)]), TARGET: np.zeros(n)}


def test_loaded_values_give_the_same_downstream_key(cache_dir):
    """A value read from the cache must hash like the freshly computed one, or every downstream entry misses once."""
    import src.compute as C
    ds = _dataset()
    for mt in (MCAR, MAR, MNAR):
        computed, loaded = C.apply_mask(ds, mt, 10, 1), C.apply_mask(ds, mt, 10, 1)
        assert np.array_equal(computed.to_numpy(), loaded.to_numpy(), equal_nan=True)
        key = lambda df: cache._entry_dir(C.apply_imputation_observed_scale, (df, KNN, 1), {})
        assert key(computed) == key(loaded), mt
        # the parallel pipeline rebuilds the DataFrame; that must not change the key either
        rebuilt = pd.DataFrame(computed.to_numpy(), columns=computed.columns.tolist(), index=computed.index)
        assert key(computed) == key(rebuilt), mt


def test_round_trip_of_every_stored_type(cache_dir):
    @cache.memory.cache
    def identity(x):
        return x

    values = [
        np.arange(6.0).reshape(2, 3),
        3.5,
        pd.DataFrame({'a': [1.0, np.nan], 'b': [2.0, 3.0]}, index=[5, 7]),
        [np.zeros((0, 2)), np.ones((2, 2))],
        [[[0.0, 1.0]], [], [[0.5, 0.7]]],                 # transform_pd output with an empty dimension
    ]
    for value in values:
        identity(value)
        loaded = identity(value)
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(loaded, value)
        elif isinstance(value, list):
            assert len(loaded) == len(value) and all(np.array_equal(np.ravel(a), np.ravel(b)) for a, b in zip(loaded, value))
        else:
            assert np.array_equal(loaded, value)


def test_empty_diagram_dimension_still_compares(cache_dir):
    """An empty dimension comes back from the cache as an array of shape (0,); the distances must accept it."""
    import gudhi
    from persim import wasserstein
    import src.compute as C
    diagram = np.array([[[0.0, 1.0, 0.0], [0.2, 0.6, 1.0]]])   # no H2 feature
    C.apply_prepare_for_comparison(diagram, PD)
    loaded = C.apply_prepare_for_comparison(diagram, PD)
    assert np.shape(loaded[2]) == (0,)
    assert gudhi.bottleneck_distance(loaded[2], np.zeros((0, 2))) == 0
    assert wasserstein(loaded[2], np.zeros((0, 2))) == 0


def _run_all(monkeypatch, workers, tmp_path):
    for m in ('src.missingness', 'src.pipeline', 'src.robustness_pipeline', 'src.comparison_metrics'):
        __import__(m)
        monkeypatch.setattr(sys.modules[m], 'SEEDS', [1, 2], raising=False)
        monkeypatch.setattr(sys.modules[m], 'WORKERS', workers, raising=False)
    import src.pipeline as P, src.robustness_pipeline as R, src.comparison_metrics as CM
    monkeypatch.chdir(tmp_path)
    ds = {'toy': _dataset()}
    P.experiment('topo', [MCAR, MAR, MNAR], [10], [MEAN, KNN, RF], [VR, DTMC, KDM], [WS, BN, L2PL, L2PI], ds)
    R.experiment_baselines('base', [MAR, MNAR], [10], [VR], [WS, L2PI], [COMPLETE_CASE, RANDOM_SUBSAMPLE, SHUFFLED_GROUND_TRUTH, MATCHED_JITTER], ds)
    CM.comparison_metrics('cmp', [MCAR, MAR], [10], [KNN], [RMSE, MAXDISP], ds)
    return {f'{f}{kind}': pd.read_csv(tmp_path / 'results' / f'{f}_{kind}.csv') for f in ('topo', 'base', 'cmp') for kind in ('results', 'distances')}


def test_second_run_is_served_from_the_cache(cache_dir, tmp_path, monkeypatch):
    """Run all three drivers twice (serial, then parallel): the second run adds no cache entry and gives the same results."""
    first = _run_all(monkeypatch, 1, tmp_path)
    entries = _entries(cache_dir)
    assert entries
    second = _run_all(monkeypatch, 2, tmp_path)
    assert _entries(cache_dir) == entries
    for name in first:
        # parallel mode collects results in completion order
        ids = [c for c in (DATASET, MISSINGNESS_TYPE, MISSING_RATE, IMPUTATION_METHOD, TDA_METHOD, DIMENSION, SEED) if c in first[name]]
        sort = lambda df: df.sort_values(ids).reset_index(drop=True)
        pd.testing.assert_frame_equal(sort(first[name]), sort(second[name]), check_like=True)


def test_per_seed_distances_match_the_aggregated_results(cache_dir, tmp_path, monkeypatch):
    """The per-seed tables hold exactly the values the result tables aggregate."""
    out = _run_all(monkeypatch, 1, tmp_path)
    for name, metric in (('topo', WS), ('base', L2PI), ('cmp', RMSE)):
        res, dist = out[f'{name}results'], out[f'{name}distances']
        ids = [c for c in (DATASET, MISSINGNESS_TYPE, MISSING_RATE, IMPUTATION_METHOD, TDA_METHOD, DIMENSION) if c in res]
        agg = dist.groupby(ids).agg(n=(SEED, 'size'), mean=(metric, 'mean')).reset_index().merge(res, on=ids)
        assert len(agg) == len(res)
        assert (agg.n == agg[N_SEEDS]).all() and np.allclose(agg['mean'], agg[metric])
