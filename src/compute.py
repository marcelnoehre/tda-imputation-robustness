from src.cache import memory
from src.tda import TDA, max_edge_length_for
from src.normalize import normalize_by_diameter
from src.missingness import MISSINGNESS
from src.imputation import IMPUTATION
from src.robustness import complete_case_analysis, random_subsample, shuffled_ground_truth
from src.downstream import downstream_score
from src.constants import FUNCTION, DATA, TARGET


@memory.cache
def get_max_edge_length(data, tda):
    return max_edge_length_for(data, tda)


@memory.cache
def apply_persistent_homology(data, tda, max_edge_length):
    return TDA[tda][FUNCTION](data, max_edge_length)


@memory.cache
def apply_missingness(dataset, mt, mr, seed):
    return MISSINGNESS[mt][FUNCTION](dataset[DATA], dataset[TARGET], mr, seed)


@memory.cache
def apply_imputation(dataset, imp, seed):
    return IMPUTATION[imp][FUNCTION](dataset, seed)


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
def apply_random_subsample(dataset_data, seed):
    return random_subsample(dataset_data, seed)


@memory.cache
def apply_shuffled_ground_truth(dataset_data, missing_df, seed):
    return shuffled_ground_truth(dataset_data, missing_df, seed)


@memory.cache
def apply_downstream_score(X, y, dataset_type, seed):
    return downstream_score(X, y, dataset_type, seed)
