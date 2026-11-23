import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _config_runtime(**env):
    code = 'from src.config import WORKERS, CONFIG; print(WORKERS, CONFIG["runtime"]["workers"])'
    clean = {k: v for k, v in os.environ.items() if k != 'TDA_WORKERS'}
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env={**clean, **env}, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_runtime_overrides_come_from_the_environment():
    """Without the override a run uses the configured worker count; main.py --workers sets it."""
    workers, configured = _config_runtime()
    assert workers == configured
    assert _config_runtime(TDA_WORKERS='3')[0] == '3'


def test_main_flags_reach_the_config():
    code = 'import sys; sys.argv = ["main.py", "--workers", "5"]; import main, src.config as c; print(c.WORKERS)'
    clean = {k: v for k, v in os.environ.items() if k != 'TDA_WORKERS'}
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=clean, capture_output=True, text=True, check=True)
    assert out.stdout.split()[-1] == '5'


def test_run_seeds_prefix():
    """--seeds replaces the global list, writes to its own result files and leaves experiments with own seeds alone."""
    import main
    from src.config import CONFIG
    configured = list(CONFIG['seeds'])
    assert main.run_seeds('baseline', None) == (configured, '')
    assert main.run_seeds('baseline', configured) == (configured, '')
    assert main.run_seeds('baseline', [1, 4, 5]) == ([1, 4, 5], 'seeds1-4-5_')
    assert main.run_seeds('mice_spread', [1, 4, 5]) == (list(CONFIG['experiments']['mice_spread']['seeds']), '')


def test_dataset_prefix_stays_a_valid_file_name():
    """Long dataset lists use their IDs: 16 names plus the experiment name exceed the 255-byte file-name limit."""
    import main
    from src.constants import DATASETS
    assert main.dataset_prefix(None, []) == ''
    assert main.dataset_prefix({DATASETS[44971]: 0, DATASETS[46762]: 0}, [44971, 46762]) == 'white_wine_quality_air_quality_and_pollution_'
    ids = [223, 666, 4353, 23516, 42367, 42369, 42999, 43384, 43402, 43437, 43623, 43695, 46764, -1, -2, -3]
    prefix = main.dataset_prefix({DATASETS[i]: 0 for i in ids}, ids)
    assert prefix.startswith('ids223_666_') and len(f'seeds1-4-5_{prefix}baseline_reference_distances.csv') < 255


def test_new_seeds_only_skips_the_first_seed_of_stochastic_settings(tmp_path, monkeypatch):
    """--new-seeds-only: seed 1 keeps the deterministic settings, the stochastic ones run only for the later seeds,
    with the same diagrams and per-seed distances as in a full run."""
    import numpy as np
    import pandas as pd
    import src.cache as cache
    import src.missingness as M
    from src.constants import DATA, TARGET, MAR, MNAR, VR, WS, BN, COMPLETE_CASE, RANDOM_SUBSAMPLE, SHUFFLED_GROUND_TRUTH

    assert M.repeated(1, MAR, True) and M.repeated(1, MNAR, True) and not M.repeated(2, MAR, True)
    monkeypatch.setattr(M, 'NEW_SEEDS_ONLY', True)
    assert M.repeated(1, MAR, True) and not M.repeated(1, MNAR, True) and not M.repeated(1, MAR, False)
    assert M.repeated(2, MNAR, True) and M.repeated(2, MAR, False) and not M.repeated(2, MAR, True)

    rng = np.random.default_rng(0)
    ds = {'toy': {DATA: pd.DataFrame(rng.normal(size=(60, 6)), columns=[f'X{i}' for i in range(6)]), TARGET: np.zeros(60)}}
    for m in ('src.missingness', 'src.pipeline', 'src.robustness_pipeline'):
        __import__(m)
        monkeypatch.setattr(sys.modules[m], 'SEEDS', [1, 2], raising=False)
        monkeypatch.setattr(sys.modules[m], 'WORKERS', 1, raising=False)
    import src.robustness_pipeline as R
    monkeypatch.chdir(tmp_path)

    def run(name, new_only):
        monkeypatch.setattr(M, 'NEW_SEEDS_ONLY', new_only)
        monkeypatch.setattr(cache, '_CACHE_DIR', str(tmp_path / name))
        R.experiment_baselines(name, [MAR, MNAR], [10], [VR], [WS, BN], [COMPLETE_CASE, RANDOM_SUBSAMPLE, SHUFFLED_GROUND_TRUTH], ds)
        root = tmp_path / name / 'src.compute' / 'apply_persistent_homology'
        return set(os.listdir(root)), pd.read_csv(tmp_path / 'results' / f'{name}_distances.csv')

    full, full_d = run('full', False)
    new, new_d = run('new', True)
    assert new < full   # strictly fewer diagrams, all of them also in the full run
    # seed 1 keeps only the deterministic pair (MAR, complete case); seed 2 is identical to the full run
    seed1 = new_d[new_d['seed'] == 1]
    assert set(zip(seed1['missingness_type'], seed1['imputation_method'])) == {(MAR, COMPLETE_CASE)}
    key = [c for c in full_d.columns if full_d[c].dtype == object] + ['seed']
    a = full_d[full_d['seed'] == 2].sort_values(key).reset_index(drop=True)
    b = new_d[new_d['seed'] == 2].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_new_seeds_only_prefix():
    code = ('import sys; sys.argv = ["main.py", "--new-seeds-only"]; import main; '
            'print(main.run_seeds("baseline", [1, 4, 5])[1])')
    clean = {k: v for k, v in os.environ.items() if k not in ('TDA_WORKERS', 'TDA_NEW_SEEDS_ONLY')}
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=clean, capture_output=True, text=True, check=True)
    assert out.stdout.split()[-1] == 'seeds4-5_'
