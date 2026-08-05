import os
import time
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, sem as scipy_sem
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.logger import log
from src.data import get_all_datasets
from src.missingness import MISSINGNESS
from src.imputation import IMPUTATION
from src.betti import estimate_betti_number
from src.utils import transform_pd
from src.constants import *
from src.compute import (
    apply_complete_case,
    apply_random_subsample,
    apply_shuffled_ground_truth,
    apply_downstream_score,
)
from src.pipeline import (
    compute_mel_dict,
    compute_original_persistence_intervals,
    normalize_original_persistence_intervals,
    prepare_original_data,
    introduce_missingness,
    impute_missing_values,
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

def _merge_method_dicts(a, b):
    '''
    Merge two {seed: {key: {mt: {mr: {method: value}}}}} dicts at the
    innermost (method) level.
    '''
    merged = {}
    for seed in set(a) | set(b):
        merged[seed] = {}
        a_key, b_key = a.get(seed, {}), b.get(seed, {})
        for key in set(a_key) | set(b_key):
            merged[seed][key] = {}
            a_mt, b_mt = a_key.get(key, {}), b_key.get(key, {})
            for mt in set(a_mt) | set(b_mt):
                merged[seed][key][mt] = {}
                a_mr, b_mr = a_mt.get(mt, {}), b_mt.get(mt, {})
                for mr in set(a_mr) | set(b_mr):
                    merged[seed][key][mt][mr] = {**a_mr.get(mr, {}), **b_mr.get(mr, {})}
    return merged

def _baseline_worker(method, dataset_data, missing_df, seed):
    '''
    Apply a baseline method to a dataset with missing values, returning the
    resulting array and the indices of the retained rows (for downstream-task
    target alignment). Used by apply_baseline_methods for parallelization.
    '''
    if method == COMPLETE_CASE:
        arr, idx = apply_complete_case(missing_df)
    elif method == RANDOM_SUBSAMPLE:
        arr, idx = apply_random_subsample(dataset_data, seed)
    else:
        arr = apply_shuffled_ground_truth(dataset_data, missing_df, seed)
        idx = np.arange(len(arr), dtype=np.float64)
    return arr, idx

def apply_baseline_methods(datasets, data_missing_values, baseline_methods):
    '''
    Apply baseline methods to all datasets with missing values, returning a
    {seed: {key: {mt: {mr: {method: array}}}} dict of the resulting arrays 
    and a parallel dict of the retained-row indices (for downstream-task
    target alignment). Parallelized across seeds/keys/mts/mrs/methods if
    WORKERS > 1.
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
    indices = {
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
                arr, idx = fut.result()
                arrays[seed][key][mt][mr][method] = arr
                indices[seed][key][mt][mr][method] = idx
    else:
        for seed, key, mt, mr, method, missing_df in tasks:
            arr, idx = _baseline_worker(method, np.array(datasets[key][DATA]), missing_df, seed)
            arrays[seed][key][mt][mr][method] = arr
            indices[seed][key][mt][mr][method] = idx

    return arrays, indices

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
    baseline_arrays, _ = apply_baseline_methods(datasets, data_missing_values, baseline_methods)
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

def _betti_worker(diagram):
    pd_diagram = transform_pd(diagram)
    return {dim: estimate_betti_number(pd_diagram[dim], add_one=(dim == 0)) for dim in DIMENSIONS}

def compute_betti_numbers(data):
    '''Mirrors pipeline.compute_distances's iteration, but estimates a
    betti number per dimension directly from each version's own diagram
    instead of computing a distance against the original.
    '''
    def _iter(data):
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    for mr, imp_dict in mr_dict.items():
                        for imp, tda_dict in imp_dict.items():
                            for tda, diagram in tda_dict.items():
                                yield seed, key, mt, mr, imp, tda, diagram

    res = {}
    tasks = list(_iter(data))

    def _place(seed, key, mt, mr, imp, tda, betti):
        (res.setdefault(seed, {}).setdefault(key, {}).setdefault(mt, {})
            .setdefault(mr, {}).setdefault(imp, {}))[tda] = {
            dim: {BETTI: v} for dim, v in betti.items()
        }

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_betti_worker, diagram): (seed, key, mt, mr, imp, tda)
                for seed, key, mt, mr, imp, tda, diagram in tasks
            }
            for fut in as_completed(futures):
                seed, key, mt, mr, imp, tda = futures[fut]
                _place(seed, key, mt, mr, imp, tda, fut.result())
    else:
        for seed, key, mt, mr, imp, tda, diagram in tasks:
            _place(seed, key, mt, mr, imp, tda, _betti_worker(diagram))

    return res

def _static_betti_stats(value):
    '''
    Returns a dictionary with static Betti number statistics.
    '''
    return {
        N_SEEDS: 1, BETTI: value, f'{BETTI}_std': 0.0, f'{BETTI}_median': value,
        f'{BETTI}_q1': value, f'{BETTI}_q3': value,
        f'{BETTI}_ci_lower': value, f'{BETTI}_ci_upper': value,
    }

def _append_static_betti_rows(filename, datasets, original_comparable, tda_methods):
    '''
    Appends an 'original' row (the estimator applied to the true,
    unperturbed data) and, where known, a 'ground_truth' row (the textbook
    Betti number) for each dataset/dimension -- static references for
    comparison, not seedwise-aggregated.
    '''
    rows = []
    for key in datasets.keys():
        for tda in tda_methods:
            orig_pd = original_comparable[key][tda][PD]
            for dim in DIMENSIONS:
                value = estimate_betti_number(orig_pd[dim], add_one=(dim == 0))
                rows.append({
                    DATASET: key, MISSINGNESS_TYPE: ORIGINAL, MISSING_RATE: 0,
                    IMPUTATION_METHOD: ORIGINAL, TDA_METHOD: tda, DIMENSION: dim,
                    **_static_betti_stats(value)
                })
            gt_betti = GROUND_TRUTH_BETTI.get(key)
            if gt_betti:
                for dim in DIMENSIONS:
                    rows.append({
                        DATASET: key, MISSINGNESS_TYPE: ORIGINAL, MISSING_RATE: 0,
                        IMPUTATION_METHOD: GROUND_TRUTH, TDA_METHOD: tda, DIMENSION: dim,
                        **_static_betti_stats(gt_betti[dim])
                    })

    path = f'results/{filename}.csv'
    existing = pd.read_csv(path)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(path, index=False)

def experiment_betti(experiment, missingness_types, missing_rates, datasets=None):
    '''
    Betti numbers for original / imputed / baseline versions, restricted
    to the manifold datasets (torus/sphere/swiss_roll) with known ground
    truth. Reuses cached apply_imputation results wherever (mt, mr, imp)
    matches an already-run server experiment (e.g. topological_metrics_
    extended at the reduced rates) -- only the 3 new baselines require new
    persistent-homology computation.
    '''
    initial_time = start_time = time.time()

    if datasets is None:
        datasets = get_all_datasets()
    datasets = {k: v for k, v in datasets.items() if k in COLLECTIONS[MANIFOLD]}
    if not datasets:
        log('No manifold datasets provided, skipping betti-number experiment')
        return
    log(f'Computing betti numbers for: {list(datasets.keys())}')

    tda_methods = [VR]

    start_time = time.time()
    mel_dict = compute_mel_dict(datasets, tda_methods)
    original_persistence_intervals = compute_original_persistence_intervals(datasets, tda_methods, mel_dict)
    normalized_original_persistence_intervals = normalize_original_persistence_intervals(original_persistence_intervals, datasets)
    original_comparable = prepare_original_data(normalized_original_persistence_intervals, [PD])
    log(f'Prepared original data in {time.time() - start_time:.2f} seconds')

    start_time = time.time()
    data_missing_values = introduce_missingness(datasets, missingness_types, missing_rates)
    log(f'Introduced missingness in {time.time() - start_time:.2f} seconds')

    start_time = time.time()
    imputation_methods = COLLECTIONS[IMPUTATION_METHOD]
    imputed_data = impute_missing_values(data_missing_values, imputation_methods)
    log(f'Imputed missing values in {time.time() - start_time:.2f} seconds (cache hits expected)')

    start_time = time.time()
    baseline_methods = COLLECTIONS[BASELINE_METHOD]
    baseline_arrays, _ = apply_baseline_methods(datasets, data_missing_values, baseline_methods)
    baseline_arrays = _filter_tiny_baselines(baseline_arrays)
    log(f'Computed baseline methods in {time.time() - start_time:.2f} seconds')

    merged = _merge_method_dicts(imputed_data, baseline_arrays)
    determinism = {**IMPUTATION, **_baseline_determinism()}

    start_time = time.time()
    persistence_intervals = compute_persistence_intervals(merged, tda_methods, determinism=determinism)
    normalized_persistence_intervals = normalize_persistence_intervals(persistence_intervals, datasets, determinism=determinism)
    log(f'Computed persistence intervals in {time.time() - start_time:.2f} seconds')

    start_time = time.time()
    betti_data = compute_betti_numbers(normalized_persistence_intervals)
    results = compute_seedwise_statistics(betti_data, determinism=determinism)
    log(f'Estimated betti numbers in {time.time() - start_time:.2f} seconds')

    os.makedirs('results', exist_ok=True)
    filename = f'{experiment}_results'
    store_results(results, filename)
    _append_static_betti_rows(filename, datasets, original_comparable, tda_methods)
    log(f'Experiment {experiment} completed in {time.time() - initial_time:.2f} seconds')

def compute_downstream_scores(datasets, merged_arrays, baseline_indices):
    '''
    Compute downstream-task predictive utility (raw features -> existing
    TARGET) for non-manifold datasets, computed for all real imputation
    methods and the 3 new baselines. Returns a {seed: {key: {mt: {mr:
    {method: score}}}}} dict of the resulting scores. Parallelized across
    seeds/keys/mts/mrs/methods if WORKERS > 1.
    '''
    def _iter(merged_arrays):
        for seed, key_dict in merged_arrays.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]:
                        for mr, method_dict in mr_dict.items():
                            for method, arr in method_dict.items():
                                yield seed, key, mt, mr, method, arr

    res = {
        seed: {
            key: {
                mt: {mr: {} for mr in mr_dict.keys()} for mt, mr_dict in mt_dict.items()
            } for key, mt_dict in key_dict.items()
        } for seed, key_dict in merged_arrays.items()
    }

    def _target_for(key, seed, mt, mr, method):
        target = np.asarray(datasets[key][TARGET])
        idx = baseline_indices.get(seed, {}).get(key, {}).get(mt, {}).get(mr, {}).get(method)
        return target[idx.astype(int)] if idx is not None else target

    tasks = list(_iter(merged_arrays))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {}
            for seed, key, mt, mr, method, arr in tasks:
                y = _target_for(key, seed, mt, mr, method)
                dataset_type = DATASET_TYPE_MAP.get(key, CLASSIFICATION)
                fut = executor.submit(apply_downstream_score, arr, y, dataset_type, seed)
                futures[fut] = (seed, key, mt, mr, method)
            for fut in as_completed(futures):
                seed, key, mt, mr, method = futures[fut]
                res[seed][key][mt][mr][method] = fut.result()
    else:
        for seed, key, mt, mr, method, arr in tasks:
            y = _target_for(key, seed, mt, mr, method)
            dataset_type = DATASET_TYPE_MAP.get(key, CLASSIFICATION)
            res[seed][key][mt][mr][method] = apply_downstream_score(arr, y, dataset_type, seed)

    return res

def _downstream_seedwise_statistics(data, determinism):
    '''
    Compute seedwise statistics for downstream-task predictive utility
    (raw features -> existing TARGET) for non-manifold datasets, computed
    for all real imputation methods and the 3 new baselines. Returns a
    {key: {mt: {mr: {method: stats}}}} dict of the resulting statistics.
    '''
    collections = {}
    for seed, key_dict in data.items():
        for key, mt_dict in key_dict.items():
            for mt, mr_dict in mt_dict.items():
                if not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]:
                    for mr, method_dict in mr_dict.items():
                        for method, value in method_dict.items():
                            if not determinism[method][DETERMINISTIC] or seed == SEEDS[0]:
                                collections.setdefault((key, mt, mr, method), []).append(value)

    res = {}
    for (key, mt, mr, method), values in collections.items():
        arr = np.array(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        n = len(valid)
        stats = {N_SEEDS: n}
        if n == 0:
            stats[DOWNSTREAM_SCORE] = np.nan
        else:
            mean = np.mean(valid)
            stats[DOWNSTREAM_SCORE] = mean
            stats[f'{DOWNSTREAM_SCORE}_std'] = np.std(valid, ddof=1) if n > 1 else 0.0
            stats[f'{DOWNSTREAM_SCORE}_median'] = np.median(valid)
            if n > 1:
                ci = t_dist.interval(0.95, df=n - 1, loc=mean, scale=scipy_sem(valid))
                stats[f'{DOWNSTREAM_SCORE}_ci_lower'] = ci[0]
                stats[f'{DOWNSTREAM_SCORE}_ci_upper'] = ci[1]
            else:
                stats[f'{DOWNSTREAM_SCORE}_ci_lower'] = mean
                stats[f'{DOWNSTREAM_SCORE}_ci_upper'] = mean
        res.setdefault(key, {}).setdefault(mt, {}).setdefault(mr, {})[method] = stats
    return res

def _store_downstream_results(results, filename):
    '''
    Store downstream-task predictive utility (raw features -> existing
    TARGET) for non-manifold datasets, computed for all real imputation
    methods and the 3 new baselines, as a CSV file with the (dataset,
    mt, mr, method) grain. Intended to be correlated in analysis notebooks
    against the WS/BN/L2PL/L2PI columns in the existing results files, not
    correlated by this pipeline itself.
    '''
    rows = []
    for dataset, mt_dict in results.items():
        for mt, mr_dict in mt_dict.items():
            for mr, method_dict in mr_dict.items():
                for method, stats in method_dict.items():
                    rows.append({DATASET: dataset, MISSINGNESS_TYPE: mt, MISSING_RATE: mr, IMPUTATION_METHOD: method, **stats})
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv(f'results/{filename}.csv', index=False)

def _original_downstream_rows(datasets):
    '''
    Compute the downstream scores for the original (non-imputed) datasets.
    Returns a list of rows with the (dataset, mt, mr, method) grain and the
    corresponding downstream score statistics. Intended to be appended to the
    results file created by _store_downstream_results, so that the original
    scores can be correlated in analysis notebooks against the WS/BN/L2PL/L2PI
    columns in the existing results files, not correlated by this pipeline
    itself.
    '''
    rows = []
    for key, dataset in datasets.items():
        dataset_type = DATASET_TYPE_MAP.get(key, CLASSIFICATION)
        X = np.asarray(dataset[DATA])
        y = np.asarray(dataset[TARGET])
        scores = np.array([apply_downstream_score(X, y, dataset_type, seed) for seed in SEEDS], dtype=float)
        valid = scores[~np.isnan(scores)]
        n = len(valid)
        stats = {N_SEEDS: n}
        if n == 0:
            stats[DOWNSTREAM_SCORE] = np.nan
        else:
            mean = np.mean(valid)
            stats[DOWNSTREAM_SCORE] = mean
            stats[f'{DOWNSTREAM_SCORE}_std'] = np.std(valid, ddof=1) if n > 1 else 0.0
            stats[f'{DOWNSTREAM_SCORE}_median'] = np.median(valid)
            if n > 1:
                ci = t_dist.interval(0.95, df=n - 1, loc=mean, scale=scipy_sem(valid))
                stats[f'{DOWNSTREAM_SCORE}_ci_lower'] = ci[0]
                stats[f'{DOWNSTREAM_SCORE}_ci_upper'] = ci[1]
            else:
                stats[f'{DOWNSTREAM_SCORE}_ci_lower'] = mean
                stats[f'{DOWNSTREAM_SCORE}_ci_upper'] = mean
        rows.append({DATASET: key, MISSINGNESS_TYPE: ORIGINAL, MISSING_RATE: 0, IMPUTATION_METHOD: ORIGINAL, **stats})
    return rows

def experiment_downstream(experiment, missingness_types, missing_rates, datasets=None):
    '''
    Downstream-task predictive utility (raw features -> existing TARGET)
    for non-manifold datasets, computed for original data, all real
    imputation methods, and the 3 new baselines. Stored as its own results
    file with the (dataset, mt, mr, method) grain -- intended to be
    correlated in analysis notebooks against the WS/BN/L2PL/L2PI columns in
    the existing results files, not correlated by this pipeline itself.
    '''
    initial_time = start_time = time.time()

    if datasets is None:
        datasets = get_all_datasets()
    datasets = {k: v for k, v in datasets.items() if k not in COLLECTIONS[MANIFOLD]}
    if not datasets:
        log('No non-manifold datasets provided, skipping downstream-task experiment')
        return
    log(f'Computing downstream task scores for: {list(datasets.keys())}')

    start_time = time.time()
    data_missing_values = introduce_missingness(datasets, missingness_types, missing_rates)
    log(f'Introduced missingness in {time.time() - start_time:.2f} seconds')

    start_time = time.time()
    imputation_methods = COLLECTIONS[IMPUTATION_METHOD]
    imputed_data = impute_missing_values(data_missing_values, imputation_methods)
    log(f'Imputed missing values in {time.time() - start_time:.2f} seconds (cache hits expected)')

    start_time = time.time()
    baseline_methods = COLLECTIONS[BASELINE_METHOD]
    baseline_arrays, baseline_indices = apply_baseline_methods(datasets, data_missing_values, baseline_methods)
    log(f'Computed baseline methods in {time.time() - start_time:.2f} seconds')

    merged = _merge_method_dicts(imputed_data, baseline_arrays)
    determinism = {**IMPUTATION, **_baseline_determinism()}

    start_time = time.time()
    downstream_data = compute_downstream_scores(datasets, merged, baseline_indices)
    results = _downstream_seedwise_statistics(downstream_data, determinism)
    log(f'Computed downstream scores in {time.time() - start_time:.2f} seconds')

    filename = f'{experiment}_results'
    _store_downstream_results(results, filename)

    path = f'results/{filename}.csv'
    existing = pd.read_csv(path)
    extra = pd.DataFrame(_original_downstream_rows(datasets))
    pd.concat([existing, extra], ignore_index=True).to_csv(path, index=False)

    log(f'Experiment {experiment} completed in {time.time() - initial_time:.2f} seconds')
