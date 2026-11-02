import argparse

from src.constants import *
from src.config import WORKERS, N_JOBS, CONFIG, experiments, experiment_seeds, use_seeds
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
    parser = argparse.ArgumentParser(description='Run the experiments listed in config/experiments.yaml.')
    parser.add_argument('ids', nargs='*', type=int, help='OpenML IDs (-1/-2/-3 = torus/swiss roll/sphere); default: all datasets')
    parser.add_argument('-e', '--experiments', nargs='+', choices=list(CONFIG['experiments']), metavar='NAME', help=f'run only these experiments: {", ".join(CONFIG["experiments"])}')
    args = parser.parse_args()
    datasets = None
    if args.ids:
        datasets = {}
        for id in args.ids:
            if id not in DATASETS:
                raise ValueError(f"Invalid dataset ID: {id}")
            datasets[DATASETS[id]] = preprocess(get_data(id), DATASETS[id])
    return datasets, args.experiments

def main():
    if WORKERS > 1 and N_JOBS != 1:
        raise ValueError('Parallelization of the inner loop will cause a pickle error.')
    datasets, selected = process_args()
    prefix = f'{"_".join(datasets.keys())}_' if datasets else ''
    for name, driver, kwargs, dataset_set in experiments():
        if selected is not None and name not in selected:
            continue
        use_seeds(experiment_seeds(name))
        if dataset_set is not None:
            # own dataset set; with dataset IDs on the command line, run only when named with -e
            if datasets is None or selected is not None:
                DRIVERS[driver](name, **kwargs, datasets=get_dataset_set(dataset_set))
        else:
            DRIVERS[driver](f'{prefix}{name}', **kwargs, datasets=datasets)

if __name__ == '__main__':
    main()
