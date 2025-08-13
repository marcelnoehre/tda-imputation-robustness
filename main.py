import sys

from src.constants import *
from src.comparison_metrics import comparison_metrics
from src.pipeline import experiment
from src.data import get_data, preprocess
from src.constants import *

def process_args():
    ids = [int(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else None
    if ids is None:
        return None
    else:
        datasets = {}
        for id in ids:
            if id not in DATASETS:
                raise ValueError(f"Invalid dataset ID: {id}")
            datasets[DATASETS[id]] = preprocess(get_data(id), DATASETS[id])
        return datasets

def main():
    datasets = process_args()
    prefix = f'{"_".join(datasets.keys())}_' if datasets else ''
    comparison_metrics(
        f'{prefix}comparison_metrics',
        [MCAR, MAR, MNAR],
        [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40],
        [CONSTANT, MEAN, MEDIAN, KNN, RF, MICE],
        [5, 10, 25],
        [KNN],
        [RMSE, MAE, FROBENIUS],
        datasets
    )
    experiment(
        f'{prefix}impact_missingness_types_rates',
        [MCAR, MAR, MNAR],
        [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40],
        [KNN],
        [VR],
        [WS, BN, L2PL, L2PI],
        datasets
    )
    experiment(
        f'{prefix}impact_imputation_methods',
        [MAR],
        [5, 10, 25], 
        [CONSTANT, MEAN, MEDIAN, KNN, RF, MICE], 
        [VR],
        [WS, BN, L2PL, L2PI],
        datasets
    )
    experiment(
        f'{prefix}impact_tda_methods',
        [MAR],
        [5, 10, 25],
        [KNN],
        [VR, DTMS, DTMC, KD],
        [WS, BN],
        datasets
    )

if __name__ == '__main__':
    main()