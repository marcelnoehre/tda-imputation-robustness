import os
from pathlib import Path
import yaml
from src import constants

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'experiments.yaml'

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

def resolve(names):
    """Map symbolic names from the config (e.g. 'MCAR') to the keys in src/constants.py."""
    return [getattr(constants, name) for name in names]

def _rates(value):
    """A rate set name or list, or a mapping mechanism -> set name or list (resolved to a dict keyed by mechanism)."""
    if isinstance(value, dict):
        return {getattr(constants, mt): _rates(rates) for mt, rates in value.items()}
    return list(CONFIG['missing_rates'][value]) if isinstance(value, str) else list(value)

def experiments():
    """
    Experiments in config order as (result name, driver, keyword arguments of the driver, dataset set).
    The dataset set is None for the study datasets, or the name of a generated set (e.g. 'synthetic_sweep').
    """
    res = []
    for name, spec in CONFIG['experiments'].items():
        kwargs = {}
        for arg, value in spec.items():
            if arg in ('driver', 'datasets', 'seeds'):
                continue
            if arg == 'missing_rates':
                kwargs[arg] = _rates(value)
            else:
                kwargs[arg] = resolve(value)
        res.append((name, spec['driver'], kwargs, spec.get('datasets')))
    return res

SEEDS = list(CONFIG['seeds'])

def experiment_seeds(name):
    """Seeds of an experiment: its own `seeds` entry, else the global list."""
    return list(CONFIG['experiments'][name].get('seeds', CONFIG['seeds']))

def use_seeds(seeds):
    """
    Set the seeds of the following run in place, so every module that imported SEEDS sees them (main process
    only; workers read the global list, but use at most SEEDS[0], which every seed list shares).
    """
    if seeds[0] != CONFIG['seeds'][0]:
        raise ValueError(f'the first seed must be {CONFIG["seeds"][0]}: deterministic settings run under it only')
    SEEDS[:] = seeds
# pool size of this run: main.py --workers sets TDA_WORKERS before src is imported (spawned workers inherit it)
WORKERS = int(os.environ.get('TDA_WORKERS', CONFIG['runtime']['workers']))
# main.py --new-seeds-only: the first seed runs only the deterministic settings, so `--seeds 1 4 5` computes seeds 4 and 5
# of the stochastic settings without repeating seed 1 (which another run computes)
NEW_SEEDS_ONLY = os.environ.get('TDA_NEW_SEEDS_ONLY') == '1'
# main.py --cached-only: skip every diagram that is not in the cache yet; the result files then lack those settings
CACHED_ONLY = os.environ.get('TDA_CACHED_ONLY') == '1'
N_JOBS = CONFIG['runtime']['n_jobs']
PROGRESS_INTERVAL = CONFIG['runtime']['progress_interval']
SLOW_TASK = CONFIG['runtime']['slow_task']

MANIFOLD_PARAMS = CONFIG['data']['manifolds']
SYNTHETIC_SWEEP = {**CONFIG['data']['synthetic_sweep'], 'manifolds': resolve(CONFIG['data']['synthetic_sweep']['manifolds'])}
MISSING_FEATURE_FRACTION = CONFIG['missingness']['missing_feature_fraction']
IMPUTATION_PARAMS = CONFIG['imputation']
DIMENSIONS = list(CONFIG['filtration']['homology_dimensions'])
MEL_PARAMS = CONFIG['filtration']['max_edge_length']
WEIGHTED_RIPS_PARAMS = CONFIG['filtration']['weighted_rips']
DTM_PARAMS = {'masses': {getattr(constants, name): m for name, m in CONFIG['filtration']['dtm']['masses'].items()}}
KD_PARAMS = {'bandwidths': {getattr(constants, name): h for name, h in CONFIG['filtration']['kernel_distance']['bandwidths'].items()}}
LANDSCAPE_PARAMS = CONFIG['vectorization']['landscape']
BETTI_PARAMS = CONFIG['betti']
IMAGE_PARAMS = CONFIG['vectorization']['image']
SUBSAMPLE_FRACTIONS = {getattr(constants, name): f for name, f in CONFIG['baselines']['subsample_fractions'].items()}
MIN_BASELINE_POINTS = CONFIG['baselines']['min_points']
