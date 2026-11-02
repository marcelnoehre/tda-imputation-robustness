import numpy as np
import pytest
import pandas as pd

from src.constants import *
from src.missingness import MISSINGNESS, repeated
from src.config import SEEDS


def _mask(mt, seed, global_seed):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f'X{i}' for i in range(8)])
    np.random.seed(global_seed)
    return MISSINGNESS[mt][FUNCTION](X, np.zeros(200), 10, seed).isna().to_numpy()


def test_masks_depend_only_on_the_seed():
    for mt in (MCAR, MAR, MNAR):
        assert np.array_equal(_mask(mt, 1, 0), _mask(mt, 1, 123)), mt


def test_masks_hit_the_rate_and_half_of_the_columns():
    for mt in (MCAR, MAR, MNAR):
        m = _mask(mt, 1, 0)
        assert m.mean() == 0.10 and m.any(axis=0).sum() == 4, mt


def test_mnar_columns_vary_with_the_seed():
    cols = {tuple(np.flatnonzero(_mask(MNAR, s, 0).any(axis=0))) for s in SEEDS}
    assert len(cols) > 1


def test_repeated():
    det, stoch = True, False
    assert repeated(SEEDS[0], MAR, det)
    assert not repeated(SEEDS[1], MAR, det)
    assert repeated(SEEDS[1], MAR, stoch) and repeated(SEEDS[1], MCAR, det) and repeated(SEEDS[1], MNAR, det)


def test_experiment_seeds_reach_every_module():
    """An experiment's own seed list replaces the global one in place, so modules that imported SEEDS see it."""
    import src.config as config
    import src.missingness as missingness
    import src.pipeline as pipeline
    default = config.experiment_seeds('topological_metrics')
    assert default == list(config.CONFIG['seeds'])
    spread = config.experiment_seeds('mice_spread')
    try:
        config.use_seeds(spread)
        assert missingness.SEEDS == spread and pipeline.SEEDS == spread
        with pytest.raises(ValueError):
            config.use_seeds([2, 3])
    finally:
        config.use_seeds(default)
    assert missingness.SEEDS == default
