import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.logger import log
from src.data import get_all_datasets
from src.missingness import MISSINGNESS
from src.imputation import IMPUTATION
from src.tda import TDA
from src.normalize import normalize_by_diameter
from src.metrics import METRICS
from src.constants import *

def _normalize_persistence_intervals(pd, dataset):
    return normalize_by_diameter(pd, dataset)

def _prepare_for_comparison(pd, ct):
    return TDA[ct][FUNCTION](pd)

def _apply_missingness(dataset, mt, mr, seed):
    return MISSINGNESS[mt][FUNCTION](dataset[DATA], dataset[TARGET], mr, seed)

def _apply_imputation(dataset, imp, seed):
    return IMPUTATION[imp][FUNCTION](dataset, seed)

def _apply_persistent_homology(dataset, tda):
    return TDA[tda][FUNCTION](dataset)
    
def _compare(metric, original, imputed, dim):
    return METRICS[metric][FUNCTION](original, imputed, dim)

def compute_original_persistence_intervals(datasets, tda_methods):
    res = {
        key: {} for key in datasets.keys()
    }

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for key, dataset in datasets.items():
            for tda in tda_methods:
                fut = executor.submit(_apply_persistent_homology, np.array(dataset[DATA]), tda)
                futures[fut] = (key, tda)

        for fut in as_completed(futures):
            key, tda = futures[fut]
            res[key][tda] = fut.result()

    return res

def normalize_original_persistence_intervals(original, datasets):
    res = {
        key: {
            tda: {} for tda in original[key].keys()
        } for key in original.keys()
    }

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for key, tda_dict in original.items():
            for tda, pd_dict in tda_dict.items():
                fut = executor.submit(_normalize_persistence_intervals, pd_dict, datasets[key][DATA])
                futures[fut] = (key, tda)
    
        for fut in as_completed(futures):
            key, tda = futures[fut]
            res[key][tda] = fut.result()
    
    return res

def prepare_original_data(original, comparison_types):
    res = {
        key: {
            tda: {} for tda in original[key].keys()
        } for key in original.keys()
    }

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for key, tda_dict in original.items():
            for tda in tda_dict.keys():
                for ct in comparison_types:
                    fut = executor.submit(_prepare_for_comparison, tda_dict[tda], ct)
                    futures[fut] = (key, tda, ct)

        for fut in as_completed(futures):
            key, tda, ct = futures[fut]
            res[key][tda][ct] = fut.result()

    return res

def introduce_missingness(datasets, missingness_types, missing_rates):
    res = {
        seed: {
            key: {
                mt: {} for mt in missingness_types
            } for key in datasets.keys()
        } for seed in SEEDS
    }

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for seed in SEEDS:
            for key, dataset in datasets.items():
                for mt in missingness_types:
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr in missing_rates:
                            fut = executor.submit(_apply_missingness, dataset, mt, mr, seed)
                            futures[fut] = (seed, key, mt, mr)

        for fut in as_completed(futures):
            seed, key, mt, mr = futures[fut]
            res[seed][key][mt][mr] = fut.result()

    return res

def impute_missing_values(data, imputation_methods):
    res = {
        seed: {
            key: {
                mt: {
                    mr: {} for mr in mr_dict.keys()
                } for mt, mr_dict in mt_dict.items()
            } for key, mt_dict in key_dict.items()
        } for seed, key_dict in data.items()
    }

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr, imp_dict in mr_dict.items():
                            for imp in imputation_methods:
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    fut = executor.submit(_apply_imputation, imp_dict, imp, seed)
                                    futures[fut] = (seed, key, mt, mr, imp)

        for fut in as_completed(futures):
            seed, key, mt, mr, imp = futures[fut]
            res[seed][key][mt][mr][imp] = fut.result()

    return res

def compute_persistence_intervals(data, tda_methods):
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

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr, imp_dict in mr_dict.items():
                            for imp, tda_dict in imp_dict.items():
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    for tda in tda_methods:
                                        fut = executor.submit(_apply_persistent_homology, tda_dict, tda)
                                        futures[fut] = (seed, key, mt, mr, imp, tda)

        for fut in as_completed(futures):
            seed, key, mt, mr, imp, tda = futures[fut]
            res[seed][key][mt][mr][imp][tda] = fut.result()
                            
    return res

def normalize_persistence_intervals(data, datasets):
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

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr, imp_dict in mr_dict.items():
                            for imp, tda_dict in imp_dict.items():
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    for tda, normalize_dict in tda_dict.items():
                                        fut = executor.submit(_normalize_persistence_intervals, normalize_dict, datasets[key][DATA])
                                        futures[fut] = (seed, key, mt, mr, imp, tda)

        for fut in as_completed(futures):
            seed, key, mt, mr, imp, tda = futures[fut]
            res[seed][key][mt][mr][imp][tda] = fut.result()

    return res

def compute_distances(original, data, metrics):
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

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for seed, key_dict in data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr, imp_dict in mr_dict.items():
                            for imp, tda_dict in imp_dict.items():
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    for tda, dim_dict in tda_dict.items():
                                        for dim in DIMENSIONS:
                                            for metric in metrics:
                                                fut = executor.submit(_compare, metric, original[key][tda], dim_dict, dim)
                                                futures[fut] = (seed, key, mt, mr, imp, tda, dim, metric)

        for fut in as_completed(futures):
            seed, key, mt, mr, imp, tda, dim, metric = futures[fut]
            res[seed][key][mt][mr][imp][tda][dim][metric] = fut.result()

    return res

def compute_seedwise_average(data):
    res = {
        key: {
            mt: {
                mr: {
                    imp: {
                        tda: {
                            dim: {
                                metric: 0.0 for metric in metric_dict.keys()
                            } for dim, metric_dict in dim_dict.items()
                        } for tda, dim_dict in tda_dict.items()
                    } for imp, tda_dict in imp_dict.items()
                } for mr, imp_dict in mr_dict.items()
            } for mt, mr_dict in mt_dict.items()
        } for key, mt_dict in next(iter(data.values())).items()
    }

    for seed, key_dict in data.items():
        for key, mt_dict in key_dict.items():
            for mt, mr_dict in mt_dict.items():
                if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                    for mr, imp_dict in mr_dict.items():
                        for imp, tda_dict in imp_dict.items():
                            if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                for tda, dim_dict in tda_dict.items():
                                    for dim, metric_dict in dim_dict.items():
                                        for metric, value in metric_dict.items():
                                            res[key][mt][mr][imp][tda][dim][metric] += value
                                            if seed == SEEDS[-1]:
                                                res[key][mt][mr][imp][tda][dim][metric] /= len(SEEDS)

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
    original_persistence_intervals = compute_original_persistence_intervals(datasets, tda_methods)
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
    persistence_intervals = compute_persistence_intervals(imputed_data, tda_methods)
    normalized_persistence_intervals = normalize_persistence_intervals(persistence_intervals, datasets)
    log(f'Computed persistence intervals in {time.time() - start_time:.2f} seconds')

    # Calculate distances
    log('Calculating distances...')
    start_time = time.time()
    distances = compute_distances(original_comparable, normalized_persistence_intervals, metrics)
    results = compute_seedwise_average(distances)
    log(f'Calculated distances in {time.time() - start_time:.2f} seconds')

    # Store results
    log('Storing results...')
    os.makedirs('results', exist_ok=True)
    store_results(results, f'{experiment}_results')
    log(f'Experiment {experiment} completed in {time.time() - initial_time:.2f} seconds')