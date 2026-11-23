from src.cache import memory
from src.tda import TDA, max_edge_length_for, filtration_params, betti_thresholds
from src.normalize import normalize_by_diameter
from src.missingness import MISSINGNESS
from src.imputation import impute
from src.robustness import complete_case_analysis, random_subsample, shuffled_ground_truth, matched_jitter
from src.data import fresh_sample
from src.constants import FUNCTION, DATA, TARGET


# Renamed from get_max_edge_length when the cutoff rule of the weighted filtrations changed, so that
# stale cache entries are not reused.
@memory.cache
def get_filtration_cutoff(data, tda):
    return max_edge_length_for(data, tda)


@memory.cache
def apply_persistent_homology(data, tda, max_edge_length, **params):
    return TDA[tda][FUNCTION](data, max_edge_length, **params)


def get_filtration_params(data, tda):
    return filtration_params(data, tda)


# Renamed from apply_missingness when the MNAR column choice became seeded, so that the unseeded MNAR masks in
# the cache are not reused (MCAR/MAR masks are recomputed identically, so their downstream entries still hit).
@memory.cache
def apply_mask(dataset, mt, mr, seed):
    return MISSINGNESS[mt][FUNCTION](dataset[DATA], dataset[TARGET], mr, seed)


# Renamed from apply_imputation when imputation moved to the scale of the observed entries, so that
# imputations of the leaky complete-data scaling in the cache are not reused.
@memory.cache
def apply_imputation_observed_scale(dataset, imp, seed):
    return impute(dataset, imp, seed)


@memory.cache
def apply_normalize(pd_data, dataset_data):
    return normalize_by_diameter(pd_data, dataset_data)


@memory.cache
def apply_prepare_for_comparison(pd_data, ct):
    return TDA[ct][FUNCTION](pd_data)


@memory.cache
def apply_complete_case(missing_df):
    return complete_case_analysis(missing_df)


@memory.cache
def apply_random_subsample(dataset_data, seed, fraction):
    return random_subsample(dataset_data, seed, fraction)


@memory.cache
def apply_shuffled_ground_truth(dataset_data, missing_df, seed):
    return shuffled_ground_truth(dataset_data, missing_df, seed)


@memory.cache
def apply_matched_jitter(dataset_data, missing_df, imputed, seed):
    return matched_jitter(dataset_data, missing_df, imputed, seed)


@memory.cache
def apply_fresh_sample(key, seed):
    return fresh_sample(key, seed)


@memory.cache
def get_betti_thresholds(data, tda, max_edge_length, **params):
    return betti_thresholds(data, tda, max_edge_length, **params)
