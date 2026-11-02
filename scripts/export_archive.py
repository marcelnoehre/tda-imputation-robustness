"""
Export the archive for Zenodo: every original and reconstructed persistence diagram of the experiments in
config/experiments.yaml, the result files and the configuration. Run it after the experiments: the diagrams
are read through the disk cache (anything missing from the cache is computed, i.e. the experiments re-run).

    python scripts/export_archive.py archive/

Layout of archive/diagrams.h5 (diagrams are divided by the diameter of the original cloud, padding removed):

    /<dataset>/<filtration>/original                                     float64 (k, 3): birth, death, dimension
    /<dataset>/<filtration>/<mechanism>/<rate>/<method>/seed_<seed>      same, for each reconstruction or reference

Every /<dataset>/<filtration> group has the attributes `cutoff` (in the units before division), `diameter` and,
for the kernel distance, `bandwidth`. Mechanisms, methods and filtrations use the keys of src/constants.py; a
deterministic setting is stored under the first seed only. The archive also holds results/*.csv (the
aggregated `<experiment>_results.csv` and the per-seed `<experiment>_distances.csv`) and config/experiments.yaml.
"""
import argparse
import os
import shutil
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.constants import *
from src.config import CONFIG_PATH, experiments
from src.data import get_all_datasets, get_dataset_set
from src.imputation import IMPUTATION
from src.normalize import diameter
from src.pipeline import (
    compute_filtration_params,
    compute_original_persistence_intervals,
    normalize_original_persistence_intervals,
    introduce_missingness,
    impute_missing_values,
    compute_persistence_intervals,
    normalize_persistence_intervals,
)
from src.robustness_pipeline import apply_baseline_methods, _filter_tiny_baselines, _baseline_determinism
from src.logger import log


def _bars(diagram):
    dg = np.asarray(diagram, dtype=np.float64).reshape(-1, 3)
    return dg[dg[:, 1] > dg[:, 0]]


def _write(f, path, diagram):
    if path not in f:
        f.create_dataset(path, data=_bars(diagram), compression='gzip')


def export_experiment(f, driver, kwargs, datasets):
    tda_methods = kwargs['tda_methods']
    params = compute_filtration_params(datasets, tda_methods)
    originals = normalize_original_persistence_intervals(
        compute_original_persistence_intervals(datasets, tda_methods, params), datasets)
    for key, tda_dict in originals.items():
        for tda, diagram in tda_dict.items():
            group = f.require_group(f'{key}/{tda}')
            cutoff, extra = params[key][tda]
            group.attrs['cutoff'] = cutoff
            group.attrs['diameter'] = diameter(datasets[key][DATA])
            for name, value in extra.items():
                group.attrs[name] = value
            _write(f, f'{key}/{tda}/original', diagram)

    masks = introduce_missingness(datasets, kwargs['missingness_types'], kwargs['missing_rates'])
    if driver == 'topological':
        clouds, determinism = impute_missing_values(masks, kwargs['imputation_methods']), IMPUTATION
    else:
        determinism = _baseline_determinism()
        clouds = _filter_tiny_baselines(apply_baseline_methods(datasets, masks, kwargs['baseline_methods']))
    diagrams = normalize_persistence_intervals(
        compute_persistence_intervals(clouds, tda_methods, params, determinism), datasets, determinism)
    for seed, key_dict in diagrams.items():
        for key, mt_dict in key_dict.items():
            for mt, mr_dict in mt_dict.items():
                for mr, method_dict in mr_dict.items():
                    for method, tda_dict in method_dict.items():
                        for tda, diagram in tda_dict.items():
                            _write(f, f'{key}/{tda}/{mt}/{mr}/{method}/seed_{seed}', diagram)


def export(out, names=None, datasets=None):
    os.makedirs(out, exist_ok=True)
    study = datasets
    with h5py.File(os.path.join(out, 'diagrams.h5'), 'a') as f:
        for name, driver, kwargs, dataset_set in experiments():
            # comparison metrics have no diagrams; the spread experiment reuses those of topological_metrics
            if driver in ('comparison_metrics', 'spread') or (names is not None and name not in names):
                continue
            log(f'Exporting the diagrams of {name}...')
            if dataset_set is not None:
                export_experiment(f, driver, kwargs, get_dataset_set(dataset_set))
            else:
                study = study if study is not None else get_all_datasets()
                export_experiment(f, driver, kwargs, study)
    os.makedirs(os.path.join(out, 'results'), exist_ok=True)
    for file in sorted(os.listdir(os.path.join(ROOT, 'results'))):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(ROOT, 'results', file), os.path.join(out, 'results', file))
    shutil.copy(CONFIG_PATH, os.path.join(out, 'experiments.yaml'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', help='output directory')
    parser.add_argument('-e', '--experiments', nargs='+', metavar='NAME', help='export only these experiments')
    args = parser.parse_args()
    export(args.out, args.experiments)


if __name__ == '__main__':
    main()
