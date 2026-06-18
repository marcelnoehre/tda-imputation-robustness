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
from src.metrics import (
    METRICS,
    compute_wasserstein_distance,
    compute_bottleneck_distance,
    landscape_l2_distance,
    persistence_image_l2_distance,
)
from src.tda import persistence_landscape, persistence_image
from src.utils import transform_pd
from src.constants import *
from src.compute import (
    get_max_edge_length,
    apply_persistent_homology,
    apply_missingness,
    apply_imputation,
    apply_normalize,
    apply_prepare_for_comparison,
)
    
def _compare_all(metrics, orig, imputed):
    pd_Y = transform_pd(imputed) if WS in metrics or BN in metrics else None
    pl_Y = persistence_landscape(imputed) if L2PL in metrics else None
    pi_Y = persistence_image(imputed) if L2PI in metrics else None
    results = {dim: {} for dim in DIMENSIONS}
    for dim in DIMENSIONS:
        for metric in metrics:
            if metric == WS:
                results[dim][metric] = compute_wasserstein_distance(orig[PD][dim], pd_Y[dim])
            elif metric == BN:
                results[dim][metric] = compute_bottleneck_distance(orig[PD][dim], pd_Y[dim])
            elif metric == L2PL:
                results[dim][metric] = landscape_l2_distance(orig[PL][dim], pl_Y[dim], orig[PD])
            elif metric == L2PI:
                results[dim][metric] = persistence_image_l2_distance(orig[PI][dim], pi_Y[dim])
            else:
                results[dim][metric] = METRICS[metric][FUNCTION](orig, imputed, dim)
    return results

def _ph_worker(data, tda, max_edge_length):
    return apply_persistent_homology(data, tda, max_edge_length)

def _normalize_worker(pd_data, dataset_data):
    return apply_normalize(pd_data, dataset_data)

def _prepare_worker(pd_data, ct):
    return apply_prepare_for_comparison(pd_data, ct)

def _missingness_worker(dataset, mt, mr, seed):
    return apply_missingness(dataset, mt, mr, seed)

def _imputation_worker(dataset, imp, seed):
    return apply_imputation(dataset, imp, seed)

def compute_mel_dict(datasets, tda_methods):
    return {
        key: {
            tda: get_max_edge_length(np.array(dataset[DATA]), tda)
            for tda in tda_methods
        }
        for key, dataset in datasets.items()
    }

def compute_original_persistence_intervals(datasets, tda_methods, mel_dict):
    def _iter(datasets, tda_methods):
        for key, dataset in datasets.items():
            for tda in tda_methods:
                yield key, tda, dataset

    res = {
        key: {} for key in datasets.keys()
    }

    tasks = list(_iter(datasets, tda_methods))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_ph_worker, np.array(dataset[DATA]), tda, mel_dict[key][tda]): (key, tda)
                for key, tda, dataset in tasks
            }
            for fut in as_completed(futures):
                key, tda = futures[fut]
                res[key][tda] = fut.result()
    else:
        for key, tda, dataset in tasks:
            res[key][tda] = apply_persistent_homology(np.array(dataset[DATA]), tda, mel_dict[key][tda])

    return res

def normalize_original_persistence_intervals(original, datasets):
    def _iter(original):
        for key, tda_dict in original.items():
            for tda, pd_dict in tda_dict.items():
                yield key, tda, pd_dict

    res = {
        key: {
            tda: {} for tda in original[key].keys()
        } for key in original.keys()
    }

    tasks = list(_iter(original))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_normalize_worker, pd_dict, datasets[key][DATA]): (key, tda)
                for key, tda, pd_dict in tasks
            }
            for fut in as_completed(futures):
                key, tda = futures[fut]
                res[key][tda] = fut.result()
    else:
        for key, tda, pd_dict in tasks:
            res[key][tda] = apply_normalize(pd_dict, datasets[key][DATA])
    
    return res

def prepare_original_data(original, comparison_types):
    def _iter(original, comparison_types):
        for key, tda_dict in original.items():
            for tda, pd_dict in tda_dict.items():
                for ct in comparison_types:
                    yield key, tda, tda_dict, ct

    res = {
        key: {
            tda: {} for tda in original[key].keys()
        } for key in original.keys()
    }

    tasks = list(_iter(original, comparison_types))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_prepare_worker, tda_dict[tda], ct): (key, tda, ct)
                for key, tda, tda_dict, ct in tasks
            }
            for fut in as_completed(futures):
                key, tda, ct = futures[fut]
                res[key][tda][ct] = fut.result()
    else:
        for key, tda, tda_dict, ct in tasks:
            res[key][tda][ct] = apply_prepare_for_comparison(tda_dict[tda], ct)

    return res

def introduce_missingness(datasets, missingness_types, missing_rates):
    def _iter(datasets, missingness_types, missing_rates):
        for seed in SEEDS:
            for key, dataset in datasets.items():
                for mt in missingness_types:
                    if not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]:
                        for mr in missing_rates:
                            yield seed, key, mt, mr, dataset

    res = {
        seed: {
            key: {
                mt: {} for mt in missingness_types
            } for key in datasets.keys()
        } for seed in SEEDS
    }

    tasks = list(_iter(datasets, missingness_types, missing_rates))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_missingness_worker, dataset, mt, mr, seed): (seed, key, mt, mr)
                for seed, key, mt, mr, dataset in tasks
            }
            for fut in as_completed(futures):
                seed, key, mt, mr = futures[fut]
                res[seed][key][mt][mr] = fut.result()
    else:
        for seed, key, mt, mr, dataset in tasks:
            res[seed][key][mt][mr] = apply_missingness(dataset, mt, mr, seed)

    return res

def impute_missing_values(data, imputation_methods):
    def _iter(data, imputation_methods):
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]:
                        for mr, imp_dict in mr_dict.items():
                            for imp in imputation_methods:
                                if not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]:
                                    yield seed, key, mt, mr, imp, imp_dict

    res = {
        seed: {
            key: {
                mt: {
                    mr: {} for mr in mr_dict.keys()
                } for mt, mr_dict in mt_dict.items()
            } for key, mt_dict in key_dict.items()
        } for seed, key_dict in data.items()
    }

    tasks = list(_iter(data, imputation_methods))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_imputation_worker, imp_dict, imp, seed): (seed, key, mt, mr, imp)
                for seed, key, mt, mr, imp, imp_dict in tasks
            }
            for fut in as_completed(futures):
                seed, key, mt, mr, imp = futures[fut]
                res[seed][key][mt][mr][imp] = fut.result()
    else:
        for seed, key, mt, mr, imp, imp_dict in tasks:
            res[seed][key][mt][mr][imp] = apply_imputation(imp_dict, imp, seed)

    return res

def compute_persistence_intervals(data, tda_methods, mel_dict):
    def _iter(data, tda_methods):
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr, imp_dict in mr_dict.items():
                            for imp, tda_dict in imp_dict.items():
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    for tda in tda_methods:
                                        yield seed, key, mt, mr, imp, tda, tda_dict

    res = {
        seed: {
            key: {
                mt: {
                    mr: {
                        imp: {} for imp in imp_dict.keys()
                    } for mr, imp_dict in mr_dict.items()
                } for mt, mr_dict in mt_dict.items()
            } for key, mt_dict in key_dict.items()
        } for seed, key_dict in data.items()
    }

    tasks = list(_iter(data, tda_methods))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_ph_worker, tda_dict, tda, mel_dict[key][tda]): (seed, key, mt, mr, imp, tda)
                for seed, key, mt, mr, imp, tda, tda_dict in tasks
            }
            for fut in as_completed(futures):
                seed, key, mt, mr, imp, tda = futures[fut]
                res[seed][key][mt][mr][imp][tda] = fut.result()
    else:
        for seed, key, mt, mr, imp, tda, tda_dict in tasks:
            res[seed][key][mt][mr][imp][tda] = apply_persistent_homology(tda_dict, tda, mel_dict[key][tda])

    return res

def normalize_persistence_intervals(data, datasets):
    def _iter(data):
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr, imp_dict in mr_dict.items():
                            for imp, tda_dict in imp_dict.items():
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    for tda, normalize_dict in tda_dict.items():
                                        yield seed, key, mt, mr, imp, tda, normalize_dict

    res = {
        seed: {
            key: {
                mt: {
                    mr: {
                        imp: {} for imp in imp_dict.keys()
                    } for mr, imp_dict in mr_dict.items()
                } for mt, mr_dict in mt_dict.items()
            } for key, mt_dict in key_dict.items()
        } for seed, key_dict in data.items()
    }

    tasks = list(_iter(data))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_normalize_worker, normalize_dict, datasets[key][DATA]): (seed, key, mt, mr, imp, tda)
                for seed, key, mt, mr, imp, tda, normalize_dict in tasks
            }
            for fut in as_completed(futures):
                seed, key, mt, mr, imp, tda = futures[fut]
                res[seed][key][mt][mr][imp][tda] = fut.result()
    else:
        for seed, key, mt, mr, imp, tda, normalize_dict in tasks:
            res[seed][key][mt][mr][imp][tda] = apply_normalize(normalize_dict, datasets[key][DATA])

    return res

def compute_distances(original, data, metrics):
    def _iter(data):
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr, imp_dict in mr_dict.items():
                            for imp, tda_dict in imp_dict.items():
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    for tda, dim_dict in tda_dict.items():
                                        yield seed, key, mt, mr, imp, tda, dim_dict

    res = {
        seed: {
            key: {
                mt: {
                    mr: {
                        imp: {
                            tda: {
                                dim: {} for dim in DIMENSIONS
                            } for tda in tda_dict.keys()
                        } for imp, tda_dict in imp_dict.items()
                    } for mr, imp_dict in mr_dict.items()
                } for mt, mr_dict in mt_dict.items()
            } for key, mt_dict in key_dict.items()
        } for seed, key_dict in data.items()
    }

    tasks = list(_iter(data))

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_compare_all, metrics, original[key][tda], dim_dict): (seed, key, mt, mr, imp, tda)
                for seed, key, mt, mr, imp, tda, dim_dict in tasks
            }
            for fut in as_completed(futures):
                seed, key, mt, mr, imp, tda = futures[fut]
                for dim, metric_dict in fut.result().items():
                    for metric, value in metric_dict.items():
                        res[seed][key][mt][mr][imp][tda][dim][metric] = value
    else:
        for seed, key, mt, mr, imp, tda, dim_dict in tasks:
            for dim, metric_dict in _compare_all(metrics, original[key][tda], dim_dict).items():
                for metric, value in metric_dict.items():
                    res[seed][key][mt][mr][imp][tda][dim][metric] = value

    return res

def compute_seedwise_statistics(data):
    collections = {}
    for seed, key_dict in data.items():
        for key, mt_dict in key_dict.items():
            for mt, mr_dict in mt_dict.items():
                if not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]:
                    for mr, imp_dict in mr_dict.items():
                        for imp, tda_dict in imp_dict.items():
                            if not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]:
                                for tda, dim_dict in tda_dict.items():
                                    for dim, metric_dict in dim_dict.items():
                                        k = (key, mt, mr, imp, tda, dim)
                                        if k not in collections:
                                            collections[k] = {m: [] for m in metric_dict}
                                        for metric, value in metric_dict.items():
                                            collections[k][metric].append(value)

    res = {}
    for (key, mt, mr, imp, tda, dim), metric_collections in collections.items():
        n = len(next(iter(metric_collections.values())))
        stats = {N_SEEDS: n}
        for metric, values in metric_collections.items():
            arr = np.array(values)
            mean = np.mean(arr)
            stats[metric] = mean
            stats[f'{metric}_std'] = np.std(arr, ddof=1) if n > 1 else np.nan
            stats[f'{metric}_median'] = np.median(arr)
            stats[f'{metric}_q1'] = np.percentile(arr, 25)
            stats[f'{metric}_q3'] = np.percentile(arr, 75)
            if n > 1:
                ci = t_dist.interval(0.95, df=n - 1, loc=mean, scale=scipy_sem(arr))
                stats[f'{metric}_ci_lower'] = ci[0]
                stats[f'{metric}_ci_upper'] = ci[1]
            else:
                stats[f'{metric}_ci_lower'] = np.nan
                stats[f'{metric}_ci_upper'] = np.nan
        res.setdefault(key, {}).setdefault(mt, {}).setdefault(mr, {}).setdefault(imp, {}).setdefault(tda, {})[dim] = stats

    return res

def store_results(results, filename):
    rows = []
    for dataset, missingness_types in results.items():
        for mt, missing_rates in missingness_types.items():
            for mr, imputations in missing_rates.items():
                for imp, tda_methods in imputations.items():
                    for tda, dimensions in tda_methods.items():
                        for dim, metrics in dimensions.items():
                            row = {
                                DATASET: dataset,
                                MISSINGNESS_TYPE: mt,
                                MISSING_RATE: mr,
                                IMPUTATION_METHOD: imp,
                                TDA_METHOD: tda,
                                DIMENSION: dim
                            }
                            for metric in metrics:
                                row[metric] = metrics[metric]

                            rows.append(row)

    pd.DataFrame(rows).to_csv(f'results/{filename}.csv', index=False)

def experiment(experiment, missingness_types, missing_rates, imputation_methods, tda_methods, metrics, datasets = None):
    initial_time = start_time = time.time()

    # Load datasets
    if datasets is None:
        log('Loading all datasets...')
        datasets = get_all_datasets()
        log(f'Loaded {len(datasets)} datasets in {time.time() - start_time:.2f} seconds')
    else:
        log(f'Loading provided datasets: {list(datasets.keys())}')

    # Original datasets
    log('Preparing original datasets...')
    start_time = time.time()
    mel_dict = compute_mel_dict(datasets, tda_methods)
    original_persistence_intervals = compute_original_persistence_intervals(datasets, tda_methods, mel_dict)
    normalized_original_persistence_intervals = normalize_original_persistence_intervals(original_persistence_intervals, datasets)
    comparisons = [{WS: PD, BN: PD, L2PL: PL, L2PI: PI}.get(metric, '_') for metric in metrics]
    original_comparable = prepare_original_data(normalized_original_persistence_intervals, comparisons)
    log(f'Prepared original data in {time.time() - start_time:.2f} seconds')

    # Introduce missingness
    log('Introducing missingness...')
    start_time = time.time()
    data_missing_values = introduce_missingness(datasets, missingness_types, missing_rates)
    log(f'Introduced missingness in {time.time() - start_time:.2f} seconds')

    # Impute missing values
    log('Imputing missing values...')
    start_time = time.time()
    imputed_data = impute_missing_values(data_missing_values, imputation_methods)
    log(f'Imputed missing values in {time.time() - start_time:.2f} seconds')

    # Compute persistence intervals
    log('Computing persistence intervals...')
    start_time = time.time()
    persistence_intervals = compute_persistence_intervals(imputed_data, tda_methods, mel_dict)
    normalized_persistence_intervals = normalize_persistence_intervals(persistence_intervals, datasets)
    log(f'Computed persistence intervals in {time.time() - start_time:.2f} seconds')

    # Calculate distances
    log('Calculating distances...')
    start_time = time.time()
    distances = compute_distances(original_comparable, normalized_persistence_intervals, metrics)
    results = compute_seedwise_statistics(distances)
    log(f'Calculated distances in {time.time() - start_time:.2f} seconds')

    # Store results
    log('Storing results...')
    os.makedirs('results', exist_ok=True)
    store_results(results, f'{experiment}_results')
    log(f'Experiment {experiment} completed in {time.time() - initial_time:.2f} seconds')