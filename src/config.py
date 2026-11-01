from pathlib import Path
import yaml
from src import constants

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'experiments.yaml'

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

def resolve(names):
    """Map symbolic names from the config (e.g. 'MCAR') to the keys in src/constants.py."""
    return [getattr(constants, name) for name in names]

def experiments():
    """Experiments in config order as (result name, driver, keyword arguments of the driver)."""
    res = []
    for name, spec in CONFIG['experiments'].items():
        kwargs = {}
        for arg, value in spec.items():
            if arg == 'driver':
                continue
            if arg == 'missing_rates':
                kwargs[arg] = CONFIG['missing_rates'][value] if isinstance(value, str) else value
            else:
                kwargs[arg] = resolve(value)
        res.append((name, spec['driver'], kwargs))
    return res

SEEDS = list(CONFIG['seeds'])
WORKERS = CONFIG['runtime']['workers']
N_JOBS = CONFIG['runtime']['n_jobs']

MANIFOLD_PARAMS = CONFIG['data']['manifolds']
MISSING_FEATURE_FRACTION = CONFIG['missingness']['missing_feature_fraction']
IMPUTATION_PARAMS = CONFIG['imputation']
DIMENSIONS = list(CONFIG['filtration']['homology_dimensions'])
MEL_PARAMS = CONFIG['filtration']['max_edge_length']
WEIGHTED_RIPS_PARAMS = CONFIG['filtration']['weighted_rips']
DTM_PARAMS = CONFIG['filtration']['dtm']
KD_PARAMS = {'bandwidths': {getattr(constants, name): h for name, h in CONFIG['filtration']['kernel_distance']['bandwidths'].items()}}
LANDSCAPE_PARAMS = CONFIG['vectorization']['landscape']
IMAGE_PARAMS = CONFIG['vectorization']['image']
SUBSAMPLE_FRACTIONS = {getattr(constants, name): f for name, f in CONFIG['baselines']['subsample_fractions'].items()}
MIN_BASELINE_POINTS = CONFIG['baselines']['min_points']
