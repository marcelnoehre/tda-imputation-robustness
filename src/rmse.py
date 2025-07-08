import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from src.logger import log
from src.data import get_all_datasets
from src.missingness import MISSINGNESS
from src.imputation import IMPUTATION
from src.constants import *

def _apply_missingness(dataset, mt, mr, seed):
    return MISSINGNESS[mt][FUNCTION](dataset[DATA], dataset[TARGET], mr, seed)

def _apply_imputation(dataset, imp, seed):
    return IMPUTATION[imp][FUNCTION](dataset, seed)

def _rmse(X: np.ndarray, Y: np.ndarray):
    if X.shape != Y.shape:
        raise ValueError(f'Arrays must have same shape, got {X.shape} vs {Y.shape}')
    diff = X - Y
    return np.sqrt(np.mean(diff**2))

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
                        for mr in mr_dict.keys():
                            for imp in imputation_methods:
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    if mr in reduced_missing_rates or imp in reduced_imputation_methods:
                                        fut = executor.submit(_apply_imputation, data[seed][key][mt][mr], imp, seed)
                                        futures[fut] = (seed, key, mt, mr, imp)

        for fut in as_completed(futures):
            seed, key, mt, mr, imp = futures[fut]
            res[seed][key][mt][mr][imp] = fut.result()

    return res

def compute_rmse(datasets, data):
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
                            for imp in imp_dict.keys():
                                if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                    fut = executor.submit(_rmse, np.asarray(datasets[key][DATA]), data[seed][key][mt][mr][imp])
                                    futures[fut] = (seed, key, mt, mr, imp)

        for fut in as_completed(futures):
            seed, key, mt, mr, imp = futures[fut]
            res[seed][key][mt][mr][imp][RMSE] = fut.result()

    return res

def compute_seedwise_average(imputed_data, rmse_data):
    res = {
        key: {
            mt: {
                mr: {
                    imp: {
                        RMSE: 0.0
                    } for imp in imp_dict.keys()
                } for mr, imp_dict in mr_dict.items()
            } for mt, mr_dict in mt_dict.items()
        } for key, mt_dict in next(iter(imputed_data.values())).items()
    }

    for seed, key_dict in imputed_data.items():
        for key, mt_dict in key_dict.items():
            for mt, mr_dict in mt_dict.items():
                if (not MISSINGNESS[mt][DETERMINISTIC] or seed == SEEDS[0]):
                    for mr, imp_dict in mr_dict.items():
                        for imp in imp_dict.keys():
                            if (not IMPUTATION[imp][DETERMINISTIC] or seed == SEEDS[0]):
                                res[key][mt][mr][imp][RMSE] += rmse_data[seed][key][mt][mr][imp][RMSE]
                                if seed == SEEDS[-1]:
                                    res[key][mt][mr][imp][RMSE] /= len(SEEDS)

    return res

def store_results(results):
    os.makedirs('results', exist_ok=True)
    rows = []
    for dataset, missingness_types in results.items():
        for mt, missing_rates in missingness_types.items():
            for mr, imputations in missing_rates.items():
                for imp in imputations.keys():
                    rows.append({
                        DATASET: dataset,
                        MISSINGNESS_TYPE: mt,
                        MISSING_RATE: mr,
                        IMPUTATION_METHOD: imp,
                        RMSE: results[dataset][mt][mr][imp][RMSE]
                    })
    pd.DataFrame(rows).to_csv(f'results/rmse.csv', index=False)

def rmse(MISSINGNESS_TYPES, MISSING_RATES, IMPUTATION_METHODS, REDUCED_MISSING_RATE, REDUCED_IMPUTATION_METHODS):
    initial_time = start_time = time.time()

    # Load all datasets
    log('Loading datasets...')
    datasets = get_all_datasets()
    log(f'Loaded {len(datasets)} datasets in {time.time() - start_time:.2f} seconds')

    # Introduce missingness
    log('Introducing missingness...')
    start_time = time.time()
    data_missing_values = introduce_missingness(datasets, MISSINGNESS_TYPES, MISSING_RATES)
    log(f'Introduced missingness in {time.time() - start_time:.2f} seconds')

    # Impute missing values
    log('Imputing missing values...')
    start_time = time.time()
    imputed_data = impute_missing_values(data_missing_values, IMPUTATION_METHODS, REDUCED_MISSING_RATE, REDUCED_IMPUTATION_METHODS)
    log(f'Imputed missing values in {time.time() - start_time:.2f} seconds')

    # Compute RMSE
    log('Computing RMSE...')
    start_time = time.time()
    rmse_data = compute_rmse(datasets, imputed_data)
    log(f'Computed RMSE in {time.time() - start_time:.2f} seconds')

    # Compute seedwise average RMSE
    log('Computing seedwise average RMSE...')
    start_time = time.time()
    results = compute_seedwise_average(imputed_data, rmse_data)
    log(f'Computed seedwise average RMSE in {time.time() - start_time:.2f} seconds')

    # Store results
    log('Storing results...')
    os.makedirs('results', exist_ok=True)
    store_results(results)
    log(f'RSMSE computed in {time.time() - initial_time:.2f} seconds')
