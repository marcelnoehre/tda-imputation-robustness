import argparse
import os

# --workers and --new-seeds-only must be in the environment before src is imported: the modules bind them at import,
# and spawned worker processes inherit the environment
RUNTIME = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
RUNTIME.add_argument('--workers', type=int, help='process pool size of this run (default: runtime.workers in the config)')
RUNTIME.add_argument('--new-seeds-only', action='store_true', help='with --seeds 1 4 5: compute the stochastic settings only for the seeds after the first (4 and 5), not again for seed 1')
RUNTIME.add_argument('--cached-only', action='store_true', help='skip diagrams that are not in the cache yet (logged as skipped); the result files lack those settings, so re-run without it once they are cached')
_runtime, _ = RUNTIME.parse_known_args()
if _runtime.cached_only:
    os.environ['TDA_CACHED_ONLY'] = '1'
if _runtime.workers is not None:
    os.environ['TDA_WORKERS'] = str(_runtime.workers)
if _runtime.new_seeds_only:
    os.environ['TDA_NEW_SEEDS_ONLY'] = '1'

from src.constants import *
from src.config import WORKERS, N_JOBS, NEW_SEEDS_ONLY, CONFIG, experiments, experiment_seeds, use_seeds
from src.comparison_metrics import comparison_metrics
from src.pipeline import experiment
from src.robustness_pipeline import experiment_baselines
from src.spread_pipeline import experiment_spread
from src.data import get_data, preprocess, get_dataset_set

DRIVERS = {
    'comparison_metrics': comparison_metrics,
    'topological': experiment,
    'baseline': experiment_baselines,
    'spread': experiment_spread,
}

def process_args():
    parser = argparse.ArgumentParser(description='Run the experiments listed in config/experiments.yaml.', parents=[RUNTIME])
    parser.add_argument('ids', nargs='*', type=int, help='OpenML IDs (-1/-2/-3 = torus/swiss roll/sphere); default: all datasets')
    parser.add_argument('-e', '--experiments', nargs='+', choices=list(CONFIG['experiments']), metavar='NAME', help=f'run only these experiments: {", ".join(CONFIG["experiments"])}')
    parser.add_argument('--seeds', nargs='+', type=int, help=f'seeds of this run instead of the config list (the first must be {CONFIG["seeds"][0]}); experiments with their own seeds keep them; results go to seeds<list>_<name> files')
    args = parser.parse_args()
    datasets = None
    if args.ids:
        datasets = {}
        for id in args.ids:
            if id not in DATASETS:
                raise ValueError(f"Invalid dataset ID: {id}")
            datasets[DATASETS[id]] = preprocess(get_data(id), DATASETS[id])
    return datasets, args.ids, args.experiments, args.seeds

def dataset_prefix(datasets, ids):
    """Result-file prefix of a run restricted to datasets: their names, or their IDs when the names get too long for a file name."""
    if not datasets:
        return ''
    names = '_'.join(datasets.keys())
    return f'{names}_' if len(names) <= 120 else f'ids{"_".join(map(str, ids))}_'

def run_seeds(name, seeds):
    """Seeds of one experiment in this run, and the result-file prefix when they differ from the configured ones."""
    configured = experiment_seeds(name)
    if seeds is None or 'seeds' in CONFIG['experiments'][name]:
        return configured, ''
    if NEW_SEEDS_ONLY:
        # the stochastic settings of the first seed are left out, so the files hold only the later seeds
        return list(seeds), f'seeds{"-".join(map(str, seeds[1:]))}_'
    return list(seeds), ('' if list(seeds) == configured else f'seeds{"-".join(map(str, seeds))}_')

def main():
    if WORKERS > 1 and N_JOBS != 1:
        raise ValueError('Parallelization of the inner loop will cause a pickle error.')
    datasets, ids, selected, seeds = process_args()
    if NEW_SEEDS_ONLY and (seeds is None or len(seeds) < 2):
        raise ValueError('--new-seeds-only needs --seeds with the first seed and the new ones, e.g. --seeds 1 4 5')
    prefix = dataset_prefix(datasets, ids)
    for name, driver, kwargs, dataset_set in experiments():
        if selected is not None and name not in selected:
            continue
        run, seed_prefix = run_seeds(name, seeds)
        use_seeds(run)
        if dataset_set is not None:
            # own dataset set; with dataset IDs on the command line, run only when named with -e
            if datasets is None or selected is not None:
                DRIVERS[driver](f'{seed_prefix}{name}', **kwargs, datasets=get_dataset_set(dataset_set))
        else:
            DRIVERS[driver](f'{seed_prefix}{prefix}{name}', **kwargs, datasets=datasets)

if __name__ == '__main__':
    main()
