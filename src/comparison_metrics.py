import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.logger import log
from src.data import get_all_datasets
from src.missingness import MISSINGNESS
from src.imputation import IMPUTATION
from src.metrics import METRICS
from src.constants import *

def _apply_missingness(dataset, mt, mr, seed):
    return MISSINGNESS[mt][FUNCTION](dataset[DATA], dataset[TARGET], mr, seed)

def _apply_imputation(dataset, imp, seed):
    return IMPUTATION[imp][FUNCTION](dataset, seed)

def _compare(metric, original, imputed):
    return METRICS[metric][FUNCTION](original, imputed)

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

def impute_missing_values(data, imputation_methods, reduced_missing_rates, reduced_imputation_methods):
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
                                    if mr in reduced_missing_rates or imp in reduced_imputation_methods:
                                        fut = executor.submit(_apply_imputation, imp_dict, imp, seed)
                                        futures[fut] = (seed, key, mt, mr, imp)

        for fut in as_completed(futures):
            seed, key, mt, mr, imp = futures[fut]
            res[seed][key][mt][mr][imp] = fut.result()

    return res

def compute_distances(datasets, imputed_data, metrics):
    res = {
        seed: {
            key: {
                mt: {
                    mr: {
                        imp: {} for imp in imp_dict.keys()
                    } for mr, imp_dict in mr_dict.items()
                } for mt, mr_dict in mt_dict.items()
            } for key, mt_dict in key_dict.items()
        } for seed, key_dict in imputed_data.items()
    }

    futures = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for seed, key_dict in imputed_data.items():
            for key, mt_dict in key_dict.items():
                for mt, mr_dict in mt_dict.items():
                    if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                        for mr, imp_dict in mr_dict.items():
                            for imp in imp_dict.keys():
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    for metric in metrics:
                                        fut = executor.submit(_compare, metric, np.asarray(datasets[key][DATA]), imputed_data[seed][key][mt][mr][imp])
                                        futures[fut] = (seed, key, mt, mr, imp, metric)

        for fut in as_completed(futures):
            seed, key, mt, mr, imp, metric = futures[fut]
            res[seed][key][mt][mr][imp][metric] = fut.result()

    return res

def compute_seedwise_average(distance_data):
    res = {
        key: {
            mt: {
                mr: {
                    imp: {
                        metric: 0.0 for metric in metrics
                    } for imp, metrics in imp_dict.items()
                } for mr, imp_dict in mr_dict.items()
            } for mt, mr_dict in mt_dict.items()
        } for key, mt_dict in next(iter(distance_data.values())).items()
    }

    for seed, key_dict in distance_data.items():
        for key, mt_dict in key_dict.items():
            for mt, mr_dict in mt_dict.items():
                if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                    for mr, imp_dict in mr_dict.items():
                        for imp, metrics in imp_dict.items():
                            if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                for metric in metrics:
                                    res[key][mt][mr][imp][metric] += distance_data[seed][key][mt][mr][imp][metric]
                                if seed == SEEDS[-1]:
                                    res[key][mt][mr][imp][metric] /= len(SEEDS)

    return res

def store_results(results, filename):
    os.makedirs('results', exist_ok=True)
    rows = []
    for dataset, missingness_types in results.items():
        for mt, missing_rates in missingness_types.items():
            for mr, imputations in missing_rates.items():
                for imp, metrics in imputations.items():
                    row = {
                        DATASET: dataset,
                        MISSINGNESS_TYPE: mt,
                        MISSING_RATE: mr,
                        IMPUTATION_METHOD: imp
                    }
                    for metric in metrics:
                        row[metric] = metrics[metric]

                    rows.append(row)

    pd.DataFrame(rows).to_csv(f'results/{filename}.csv', index=False)

def comparison_metrics(comparison_metrics, missingness_types, missing_rates, imputation_methods, reduced_missing_rates, reduced_imputation_methods, metrics, datasets=None):
    initial_time = start_time = time.time()

    # Load datasets
    if datasets is None:
        log('Loading all datasets...')
        datasets = get_all_datasets()
        log(f'Loaded {len(datasets)} datasets in {time.time() - start_time:.2f} seconds')
    else:
        log(f'Loading provided datasets: {list(datasets.keys())}')

    # Introduce missingness
    log('Introducing missingness...')
    start_time = time.time()
    data_missing_values = introduce_missingness(datasets, missingness_types, missing_rates)
    log(f'Introduced missingness in {time.time() - start_time:.2f} seconds')

    # Impute missing values
    log('Imputing missing values...')
    start_time = time.time()
    imputed_data = impute_missing_values(data_missing_values, imputation_methods, reduced_missing_rates, reduced_imputation_methods)
    log(f'Imputed missing values in {time.time() - start_time:.2f} seconds')

    # Compute Distances
    log('Computing Distances...')
    start_time = time.time()
    distance_data = compute_distances(datasets, imputed_data, metrics)
    log(f'Computed distances in {time.time() - start_time:.2f} seconds')

    # Compute seedwise average distances
    log('Computing seedwise average distances...')
    start_time = time.time()
    results = compute_seedwise_average(distance_data)
    log(f'Computed seedwise average distances in {time.time() - start_time:.2f} seconds')

    # Store results
    log('Storing results...')
    os.makedirs('results', exist_ok=True)
    store_results(results, f'{comparison_metrics}_results')
    log(f'Distances computed in {time.time() - initial_time:.2f} seconds')