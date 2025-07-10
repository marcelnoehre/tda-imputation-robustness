from typing import Dict

### OpenML Data ###
MIN_NUM_FEATURES = 8
MAX_NUM_FEATURES = 30
MIN_SAMPLES = 500
MAX_SAMPLES = 2000
DATASET_KEYS = ['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']
STOCK_DATASET_ID = 223
RMFTSA_LADATA_DATASET_ID = 666
CONCRETE_DATA_ID = 4353
TREASURY_DATASET_ID = 42367
WEATHER_IZMIR_DATASET_ID = 42369
HUNGARIAN_CHICKENPOX_DATASET_ID = 42999
CNN_STOCK_PRED_DJI_DATASET_ID = 43000
DIABETES_DATASET_ID = 43384
RED_WINE_QUALITY_DATASET_ID = 43695

# Torus
TORUS_SAMPLES = 1000
TORUS_AMBIENT = 20
TORUS_SEED = 42

# On an On (NCS)
AUDIO_PATH = 'assets/on_and_on_ncs.wav'
HOP_LENGTH = 512
WINDOW_DURATION = 0.5 
STEP_DURATION = 0.2
CHROMA_LABELS = ['C', 'Csharp', 'D', 'Dsharp', 'E', 'F', 'Fsharp', 'G', 'Gsharp', 'A', 'Asharp', 'B']

### DATA ###
DATA = 'data'
TARGET = 'target'
FUNCTION = 'function'
STOCK = 'stock'
RMFTSA_LADATA = 'rmftsa_ladata'
CONCRETE_DATA = 'concrete_data'
TREASURY = 'treasury'
WEATHER_IZMIR = 'weather_izmir'
HUNGARIAN_CHICKENPOX = 'hungarian-chickenpox'
CNN_STOCK_PRED_DJI = 'cnn-stock-pred-dji'
DIABETES = 'diabetes'
RED_WINE_QUALITY = 'red_wine_quality'
TORUS = 'torus'
ON_AND_ON = 'on_and_on'

### EXPERIMENTS ###
SEEDS = [42, 123, 2025]
MCAR = 'missing_completely_at_random'
MAR = 'missing_at_random'
MNAR = 'missing_not_at_random'
CONSTANT = 'constant_imputation'
MEAN = 'mean_imputation'
MEDIAN = 'median_imputation'
KNN = 'k_nearest_neighbors_imputation'
RF = 'random_forest_imputation'
MICE = 'mice_imputation'
VR = 'vietoris_rips'
DTM = 'distance_to_a_measure'
KD = 'kernel_distance'
PD = 'persistence_diagram'
PL = 'persistence_landscape'
PI = 'persistence_image'
WS = 'wasserstein_distance'
BN = 'bottleneck_distance'
L2PL = 'l2_distance_landscape'
L2PI = 'l2_distance_image'
RMSE = 'rmse'

### PIPELINE ###
WORKERS = 8
N_JOBS = 1
ESTIMATORS = 30
DEPTH = 15
ITERATIONS = 5
DIMENSIONS = [0, 1, 2]
EUCLIDEAN = 2
DETERMINISTIC = 'deterministic'

### IDENTIFIER ###
DATASET = 'dataset'
MISSINGNESS_TYPE = 'missingness_type'
MISSING_RATE = 'missing_rate'
IMPUTATION_METHOD = 'imputation_method'
TDA_METHOD = 'tda_method'
METRIC = 'metric'
DIMENSION = 'dimension'

### COLLECTIONS ###
COLLECTIONS = {
    MISSINGNESS_TYPE: [MCAR, MAR, MNAR],
    IMPUTATION_METHOD: [CONSTANT, MEAN, MEDIAN, KNN, RF, MICE],
    TDA_METHOD: [VR, DTM, KD],
    METRIC: [WS, BN, L2PL, L2PI],
}

### RESULTS ###
RMSE_RESULTS = '../results/rmse.csv'
IMPACT_MISSINGNESS = '../results/impact_missingness_types_rates_results.csv'
IMPACT_IMPUTATION = '../results/impact_imputation_methods_results.csv'
IMPACT_TDA = '../results/impact_tda_methods_results.csv'

### LABELS ###
LABEL: Dict[str, str] = {
    STOCK: 'Stock Dataset',
    RMFTSA_LADATA: 'RMFTSA LA Data',
    CONCRETE_DATA: 'Concrete Data',
    TREASURY: 'Treasury',
    WEATHER_IZMIR: 'Weather Izmir',
    HUNGARIAN_CHICKENPOX: 'Hungarian Chickenpox',
    CNN_STOCK_PRED_DJI: 'CNN Stock Prediction DJI',
    DIABETES: 'Diabetes Dataset',
    RED_WINE_QUALITY: 'Red Wine Quality',
    TORUS: 'Torus (Synthetic)',
    ON_AND_ON: 'On & On (NCS)',
    MCAR: 'Missing Completely At Random',
    MAR: 'Missing At Random',
    MNAR: 'Missing Not At Random',
    CONSTANT: 'Constant Imputation',
    MEAN: 'Mean Imputation',
    MEDIAN: 'Median Imputation',
    KNN: 'K Nearest Neighbors Imputation',
    RF: 'Random Forest Imputation',
    MICE: 'Multiple Imputation by Chained Equations',
    VR: 'Vietoris Rips Complex',
    DTM: 'Distance To a Measure',
    KD: 'Kernel Distance',
    PD: 'Persistence Diagram',
    PL: 'Persistence Landscape',
    PI: 'Persistence Image',
    WS: 'Wasserstein Distance',
    BN: 'Bottleneck Distance',
    L2PL: 'L2 Distance (Persistence Landscape)',
    L2PI: 'L2 Distance (Persistence Image)',
    RMSE: 'Root Mean Square Error',
    DATASET: 'Dataset',
    MISSINGNESS_TYPE: 'Type of Missingness',
    MISSING_RATE: 'Missing Rate [%]',
    IMPUTATION_METHOD: 'Imputation Method',
    TDA_METHOD: 'TDA Method',
    DIMENSION: 'Dimension'
}

LABEL_SHORT: Dict[str, str] = {
    MCAR: 'MCAR',
    MAR: 'MAR',
    MNAR: 'MNAR',
    DIMENSION: 'Dim'
}