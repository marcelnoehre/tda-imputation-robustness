import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.logger import log
from src.data import get_all_datasets
from src.missingness import MISSINGNESS
from src.constants import *
from src.compute import (
    apply_complete_case,
    apply_random_subsample,
    apply_shuffled_ground_truth,
)
from src.pipeline import (
    compute_mel_dict,
    compute_original_persistence_intervals,
    normalize_original_persistence_intervals,
    prepare_original_data,
    introduce_missingness,
    compute_persistence_intervals,
    normalize_persistence_intervals,
    compute_distances,
    compute_seedwise_statistics,
    store_results,
)

BASELINE_DETERMINISM = {
    COMPLETE_CASE: True,
    RANDOM_SUBSAMPLE: False,
    SHUFFLED_GROUND_TRUTH: False,
}

def _baseline_determinism():
    return {m: {DETERMINISTIC: BASELINE_DETERMINISM[m]} for m in COLLECTIONS[BASELINE_METHOD]}

def _baseline_worker(method, dataset_data, missing_df, seed):
    '''
    Apply a baseline method to a dataset with missing values, returning the
    resulting array. Used by apply_baseline_methods for parallelization.
    '''
    if method == COMPLETE_CASE:
        arr, _ = apply_complete_case(missing_df)
    elif method == RANDOM_SUBSAMPLE:
        arr, _ = apply_random_subsample(dataset_data, seed)
    else:
        arr = apply_shuffled_ground_truth(dataset_data, missing_df, seed)
    return arr

def apply_baseline_methods(datasets, data_missing_values, baseline_methods):
    '''
    Apply baseline methods to all datasets with missing values, returning a
    {seed: {key: {mt: {mr: {method: array}}}} dict of the resulting arrays.
    Parallelized across seeds/keys/mts/mrs/methods if WORKERS > 1.
    '''
    def _iter(data_missing_values, baseline_methods):
        for seed, key_dict in data_missing_values.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]:
                        for mr, missing_df in mr_dict.items():
                            for method in baseline_methods:
                                if not BASELINE_DETERMINISM[method] or seed == SEEDS[0]:
                                    yield seed, key, mt, mr, method, missing_df

    tasks = list(_iter(data_missing_values, baseline_methods))

    arrays = {
        seed: {
            key: {
                mt: {mr: {} for mr in mr_dict.keys()} for mt, mr_dict in mt_dict.items()
            } for key, mt_dict in key_dict.items()
        } for seed, key_dict in data_missing_values.items()
    }

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_baseline_worker, method, np.array(datasets[key][DATA]), missing_df, seed): (seed, key, mt, mr, method)
                for seed, key, mt, mr, method, missing_df in tasks
            }
            for fut in as_completed(futures):
                seed, key, mt, mr, method = futures[fut]
                arrays[seed][key][mt][mr][method] = fut.result()
    else:
        for seed, key, mt, mr, method, missing_df in tasks:
            arrays[seed][key][mt][mr][method] = _baseline_worker(method, np.array(datasets[key][DATA]), missing_df, seed)

    return arrays

def _filter_tiny_baselines(baseline_arrays, min_points=MIN_BASELINE_POINTS):
    '''
    Drop (seed, key, mt, mr, method) entries whose point cloud is too
    small for meaningful persistent homology (e.g. complete-case analysis at
    high missing rates can leave very few or zero rows). Logs how many were
    dropped rather than silently truncating.
    '''
    dropped = 0
    kept = {}
    for seed, key_dict in baseline_arrays.items():
        kept[seed] = {}
        for key, mt_dict in key_dict.items():
            kept[seed][key] = {}
            for mt, mr_dict in mt_dict.items():
                kept[seed][key][mt] = {}
                for mr, method_dict in mr_dict.items():
                    kept[seed][key][mt][mr] = {}
                    for method, arr in method_dict.items():
                        if len(arr) >= min_points:
                            kept[seed][key][mt][mr][method] = arr
                        else:
                            dropped += 1
    if dropped:
        log(f'Skipped {dropped} baseline combination(s) with fewer than {min_points} points remaining')
    return kept

def experiment_baselines(experiment, missingness_types, missing_rates, tda_methods, metrics, datasets=None):
    '''
    Random-subsample / complete-case / shuffled-ground-truth baselines,
    compared against original persistent homology the same way real
    imputation methods are. Uses the IMPUTATION_METHOD column/schema so
    results can be directly concatenated with impact_missingness_types_rates
    and impact_imputation_methods for comparison.
    '''
    initial_time = start_time = time.time()

    if datasets is None:
        log('Loading all datasets...')
        datasets = get_all_datasets()
        log(f'Loaded {len(datasets)} datasets in {time.time() - start_time:.2f} seconds')
    else:
        log(f'Loading provided datasets: {list(datasets.keys())}')

    log('Preparing original datasets...')
    start_time = time.time()
    mel_dict = compute_mel_dict(datasets, tda_methods)
    original_persistence_intervals = compute_original_persistence_intervals(datasets, tda_methods, mel_dict)
    normalized_original_persistence_intervals = normalize_original_persistence_intervals(original_persistence_intervals, datasets)
    comparisons = [{WS: PD, BN: PD, L2PL: PL, L2PI: PI}.get(metric, '_') for metric in metrics]
    original_comparable = prepare_original_data(normalized_original_persistence_intervals, comparisons)
    log(f'Prepared original data in {time.time() - start_time:.2f} seconds')

    log('Introducing missingness...')
    start_time = time.time()
    data_missing_values = introduce_missingness(datasets, missingness_types, missing_rates)
    log(f'Introduced missingness in {time.time() - start_time:.2f} seconds')

    log('Computing baseline methods...')
    start_time = time.time()
    baseline_methods = COLLECTIONS[BASELINE_METHOD]
    baseline_arrays = apply_baseline_methods(datasets, data_missing_values, baseline_methods)
    baseline_arrays = _filter_tiny_baselines(baseline_arrays)
    log(f'Computed baseline methods in {time.time() - start_time:.2f} seconds')

    log('Computing persistence intervals...')
    start_time = time.time()
    determinism = _baseline_determinism()
    persistence_intervals = compute_persistence_intervals(baseline_arrays, tda_methods, determinism=determinism)
    normalized_persistence_intervals = normalize_persistence_intervals(persistence_intervals, datasets, determinism=determinism)
    log(f'Computed persistence intervals in {time.time() - start_time:.2f} seconds')

    log('Calculating distances...')
    start_time = time.time()
    distances = compute_distances(original_comparable, normalized_persistence_intervals, metrics, determinism=determinism)
    results = compute_seedwise_statistics(distances, determinism=determinism)
    log(f'Calculated distances in {time.time() - start_time:.2f} seconds')

    log('Storing results...')
    os.makedirs('results', exist_ok=True)
    store_results(results, f'{experiment}_results')
    log(f'Experiment {experiment} completed in {time.time() - initial_time:.2f} seconds')
