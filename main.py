from src.constants import *
from src.comparison_metrics import comparison_metrics
from src.pipeline import experiment

def main():
    comparison_metrics(
        [MCAR, MAR, MNAR],
        [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40],
        [CONSTANT, MEAN, MEDIAN, KNN, RF, MICE],
        [5, 10, 25],
        [KNN],
        [RMSE, MAE, FROBENIUS]
    )
    experiment(
        'impact_missingness_types_rates',
        [MCAR, MAR, MNAR],
        [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40],
        [KNN],
        [VR],
        [WS, BN, L2PL, L2PI]
    )
    experiment(
        'impact_imputation_methods',
        [MAR],
        [5, 10, 25], 
        [CONSTANT, MEAN, MEDIAN, KNN, RF, MICE], 
        [VR],
        [WS, BN, L2PL, L2PI]
    )
    experiment(
        'impact_tda_methods',
        [MAR],
        [5, 10, 25],
        [KNN],
        [VR, DTM, KD],
        [WS, BN]
    )

if __name__ == '__main__':
    main()