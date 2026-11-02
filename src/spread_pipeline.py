import os
import time
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import as_completed
from src.utils import transform_pd, process_pool
from src.logger import log, timed, Progress
from src.data import get_all_datasets
from src.constants import *
from src.config import WORKERS
from src.pipeline import (
    compute_filtration_params,
    compute_original_persistence_intervals,
    normalize_original_persistence_intervals,
    prepare_original_data,
    introduce_missingness,
    impute_missing_values,
    compute_persistence_intervals,
    normalize_persistence_intervals,
    compute_distances,
    _compare_all,
)

def _pair_worker(metrics, diagram_a, diagram_b):
    '''
    Distances between the diagrams of two draws of the same imputer, in the
    same way as an imputed diagram is compared with the original one.
    '''
    return _compare_all(metrics, {PD: transform_pd(diagram_a), DIAGRAM: np.asarray(diagram_a)}, diagram_b)

def compute_pairwise_spread(diagrams, metrics):
    '''
    Pairwise distances among the diagrams of all seeds (draws) of each
    stochastic imputer. Returns rows (dataset, mechanism, rate, method,
    filtration, dimension, seed pair, metrics).
    '''
    groups = {}
    for seed, key_dict in diagrams.items():
        for key, mt_dict in key_dict.items():
            for mt, mr_dict in mt_dict.items():
                for mr, imp_dict in mr_dict.items():
                    for imp, tda_dict in imp_dict.items():
                        for tda, diagram in tda_dict.items():
                            groups.setdefault((key, mt, mr, imp, tda), {})[seed] = diagram
    tasks = [(group, a, b) for group, draws in groups.items() if len(draws) > 1
             for a, b in itertools.combinations(sorted(draws), 2)]

    rows = []
    def _store(group, a, b, result):
        for dim, values in result.items():
            rows.append({DATASET: group[0], MISSINGNESS_TYPE: group[1], MISSING_RATE: group[2], IMPUTATION_METHOD: group[3],
                         TDA_METHOD: group[4], DIMENSION: dim, 'seed_a': a, 'seed_b': b, **values})

    progress = Progress('Pairwise distances between draws', len(tasks))
    if WORKERS > 1:
        with process_pool(WORKERS) as executor:
            futures = {
                executor.submit(timed, _pair_worker, metrics, groups[group][a], groups[group][b]): (group, a, b)
                for group, a, b in tasks
            }
            for fut in as_completed(futures):
                group, a, b = futures[fut]
                _store(group, a, b, progress(fut.result(), *group, f'seeds {a}/{b}'))
    else:
        for group, a, b in tasks:
            _store(group, a, b, progress(timed(_pair_worker, metrics, groups[group][a], groups[group][b]), *group, f'seeds {a}/{b}'))
    return pd.DataFrame(rows)

def experiment_spread(experiment, missingness_types, missing_rates, imputation_methods, tda_methods, metrics, datasets=None):
    '''
    Within-imputation spread of stochastic imputers (e.g. MICE): the seeds
    are independent draws on the same mask (MAR masks are deterministic), so
    the pairwise distances among their diagrams measure how much the topology
    varies between imputations, which is compared with the distance of the
    draws to the original diagram. Writes <experiment>_pairs.csv (every
    pair) and <experiment>_results.csv (per setting: median pairwise
    distance, mean distance to the original, and their ratio).
    '''
    initial_time = time.time()
    if datasets is None:
        log('Loading all datasets...')
        datasets = get_all_datasets()
    else:
        log(f'Loading provided datasets: {list(datasets.keys())}')

    log('Preparing original datasets...')
    filtration_params = compute_filtration_params(datasets, tda_methods)
    originals = normalize_original_persistence_intervals(
        compute_original_persistence_intervals(datasets, tda_methods, filtration_params), datasets)
    comparisons = list(dict.fromkeys({WS: PD, BN: PD, L2PL: DIAGRAM, L2PI: DIAGRAM}[metric] for metric in metrics))
    original_comparable = prepare_original_data(originals, comparisons)

    log('Introducing missingness and imputing...')
    data_missing_values = introduce_missingness(datasets, missingness_types, missing_rates)
    imputed_data = impute_missing_values(data_missing_values, imputation_methods)
    diagrams = normalize_persistence_intervals(
        compute_persistence_intervals(imputed_data, tda_methods, filtration_params), datasets)

    log('Calculating distances...')
    pairs = compute_pairwise_spread(diagrams, metrics)
    topo = compute_distances(original_comparable, diagrams, metrics)
    topo_rows = [{DATASET: key, MISSINGNESS_TYPE: mt, MISSING_RATE: mr, IMPUTATION_METHOD: imp, TDA_METHOD: tda, DIMENSION: dim, **values}
                 for seed, key_dict in topo.items() for key, mt_dict in key_dict.items() for mt, mr_dict in mt_dict.items()
                 for mr, imp_dict in mr_dict.items() for imp, tda_dict in imp_dict.items() for tda, dim_dict in tda_dict.items()
                 for dim, values in dim_dict.items() if values]
    ids = [DATASET, MISSINGNESS_TYPE, MISSING_RATE, IMPUTATION_METHOD, TDA_METHOD, DIMENSION]
    spread = pairs.groupby(ids)[metrics].median().add_suffix('_spread')
    to_original = pd.DataFrame(topo_rows).groupby(ids)[metrics].mean().add_suffix('_to_original')
    results = spread.join(to_original, how='inner')
    for metric in metrics:
        results[f'{metric}_ratio'] = results[f'{metric}_spread'] / results[f'{metric}_to_original']

    os.makedirs('results', exist_ok=True)
    pairs.to_csv(f'results/{experiment}_pairs.csv', index=False)
    results.reset_index().to_csv(f'results/{experiment}_results.csv', index=False)
    log(f'Experiment {experiment} completed in {time.time() - initial_time:.2f} seconds')
