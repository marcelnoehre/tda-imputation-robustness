from typing import Dict

### OpenML Data ###
MIN_NUM_FEATURES = 8
MAX_NUM_FEATURES = 50
MIN_SAMPLES = 500
MAX_SAMPLES = 5000
DATASET_KEYS = ['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']
STOCK_DATASET_ID = 223
RMFTSA_LADATA_DATASET_ID = 666
CONCRETE_DATA_DATASET_ID = 4353
DEBUTANIZER_DATASET_ID = 23516
TREASURY_DATASET_ID = 42367
WEATHER_IZMIR_DATASET_ID = 42369
HUNGARIAN_CHICKENPOX_DATASET_ID = 42999
CNN_STOCK_PRED_DJI_DATASET_ID = 43000
DIABETES_DATASET_ID = 43384
INDIAN_STOCK_MARKET_DATASET_ID = 43402
GENDER_RECOGNITION_DATASET_ID = 43437
BOSTON_WEATHER_DATASET_ID = 43623
RED_WINE_QUALITY_DATASET_ID = 43695
WHITE_WINE_QUALITY_DATASET_ID = 44971
AIR_QUALITY_DATASET_ID = 46762
FOOTBALL_PLAYER_POSITION_DATASET_ID = 46764

# Manifold Data
MANIFOLD_SAMPLES = 1000
MANIFOLD_AMBIENT = 20
MANIFOLD_SEED = 42

# On an On (NCS)
AUDIO_PATH = 'assets/on_and_on_ncs.wav'
HOP_LENGTH = 512
WINDOW_DURATION = 0.5 
STEP_DURATION = 0.2
CHROMA_LABELS = ['C', 'Csharp', 'D', 'Dsharp', 'E', 'F', 'Fsharp', 'G', 'Gsharp', 'A', 'Asharp', 'B']

### DATA ###
MANIFOLD = 'manifold'
TIME_SERIES = 'time_series'
REGRESSION = 'regression'
CLASSIFICATION = 'classification'

DATA = 'data'
TARGET = 'target'
FUNCTION = 'function'
STOCK = 'stock'
RMFTSA_LADATA = 'rmftsa_ladata'
CONCRETE_DATA = 'concrete_data'
DEBUTANIZER = 'debutanizer'
TREASURY = 'treasury'
WEATHER_IZMIR = 'weather_izmir'
HUNGARIAN_CHICKENPOX = 'hungarian_chickenpox'
CNN_STOCK_PRED_DJI = 'cnn_stock_pred_dji'
DIABETES = 'diabetes'
INDIAN_STOCK_MARKET = 'indian_stock_market'
GENDER_RECOGNITION = 'gender_recognition_by_voice'
BOSTON_WEATHER = 'boston_weather_data'
RED_WINE_QUALITY = 'red_wine_quality'
WHITE_WINE_QUALITY = 'white_wine_quality'
AIR_QUALITY = 'air_quality_and_pollution'
FOOTBALL_PLAYER_POSITION = 'football_player_position'
TORUS = 'torus'
SWISS_ROLL = 'swiss_roll'
SPHERE = 'sphere'
ON_AND_ON = 'on_and_on'

DATASETS = {
    STOCK_DATASET_ID: STOCK,
    RMFTSA_LADATA_DATASET_ID: RMFTSA_LADATA,
    CONCRETE_DATA_DATASET_ID: CONCRETE_DATA,
    DEBUTANIZER_DATASET_ID: DEBUTANIZER,
    TREASURY_DATASET_ID: TREASURY,
    WEATHER_IZMIR_DATASET_ID: WEATHER_IZMIR,
    HUNGARIAN_CHICKENPOX_DATASET_ID: HUNGARIAN_CHICKENPOX,
    CNN_STOCK_PRED_DJI_DATASET_ID: CNN_STOCK_PRED_DJI,
    DIABETES_DATASET_ID: DIABETES,
    INDIAN_STOCK_MARKET_DATASET_ID: INDIAN_STOCK_MARKET,
    GENDER_RECOGNITION_DATASET_ID: GENDER_RECOGNITION,
    BOSTON_WEATHER_DATASET_ID: BOSTON_WEATHER,
    RED_WINE_QUALITY_DATASET_ID: RED_WINE_QUALITY,
    WHITE_WINE_QUALITY_DATASET_ID: WHITE_WINE_QUALITY,
    AIR_QUALITY_DATASET_ID: AIR_QUALITY,
    FOOTBALL_PLAYER_POSITION_DATASET_ID: FOOTBALL_PLAYER_POSITION,
    -1: TORUS,
    -2: SWISS_ROLL,
    -3: SPHERE,
    -4: ON_AND_ON
}

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
DTMS = 'distance_to_a_measure_sqrt'
DTMC = 'distance_to_a_measure_chazal'
KD = 'kernel_distance'
PD = 'persistence_diagram'
PL = 'persistence_landscape'
PI = 'persistence_image'
WS = 'wasserstein_distance'
BN = 'bottleneck_distance'
L2PL = 'l2_distance_landscape'
L2PI = 'l2_distance_image'
RMSE = 'root_mean_square_error'
MAE = 'mean_absolute_error'

TDA_ERROR = 'tda_error'
COMP_ERROR = 'comparison_error'

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
DATASET_TYPE = 'dataset_type'
MISSINGNESS_TYPE = 'missingness_type'
MISSING_RATE = 'missing_rate'
IMPUTATION_METHOD = 'imputation_method'
TDA_METHOD = 'tda_method'
METRIC = 'metric'
TDA_METRIC = 'tda_metric'
TDA_METRIC_SHORT = 'tda_metric_short'
DIMENSION = 'dimension'
DISTANCE = 'distance'
SCALED_DISTANCE = 'scaled_distance'
NORMALIZED_DISTANCE = 'normalized_distance'
RESULT = 'result'
LINESTYLE = 'linestyle'

### RESULTS ###
COMPARISON_METRICS = '../results/comparison_metrics_results.csv'
IMPACT_MISSINGNESS = '../results/impact_missingness_types_rates_results.csv'
IMPACT_IMPUTATION = '../results/impact_imputation_methods_results.csv'
IMPACT_TDA = '../results/impact_tda_methods_results.csv'

### VISUALIZATION ###
HEIGHT = 4.5
ASPECT = 1.4

COLOR_MAP = {
    DIMENSION: 'Set1',
    MISSINGNESS_TYPE: 'Set2',
    IMPUTATION_METHOD: 'tab10',
    TDA_METHOD: 'Dark2',
}

### COLLECTIONS ###
COLLECTIONS = {
    DATA: [AIR_QUALITY, BOSTON_WEATHER, CNN_STOCK_PRED_DJI, CONCRETE_DATA, DEBUTANIZER,
           DIABETES, FOOTBALL_PLAYER_POSITION, GENDER_RECOGNITION, HUNGARIAN_CHICKENPOX,
           INDIAN_STOCK_MARKET, ON_AND_ON, RED_WINE_QUALITY, RMFTSA_LADATA, SPHERE,
           STOCK, SWISS_ROLL, TORUS, TREASURY, WEATHER_IZMIR, WHITE_WINE_QUALITY],
    DATASET_TYPE: [MANIFOLD, TIME_SERIES, REGRESSION, CLASSIFICATION],
    METRIC: [MAE, RMSE],
    MISSINGNESS_TYPE: [MCAR, MAR, MNAR],
    MISSING_RATE: [5, 10, 25],
    IMPUTATION_METHOD: [CONSTANT, MEAN, MEDIAN, KNN, RF, MICE],
    TDA_METHOD: [VR, DTMS, DTMC, KD],
    TDA_METRIC: [WS, BN, L2PL, L2PI],
    TDA_METRIC_SHORT: [WS, BN],
    RESULT: [COMPARISON_METRICS, IMPACT_MISSINGNESS, IMPACT_IMPUTATION, IMPACT_TDA],
    MANIFOLD: [SPHERE, SWISS_ROLL, TORUS],
    TIME_SERIES: [BOSTON_WEATHER, CNN_STOCK_PRED_DJI, HUNGARIAN_CHICKENPOX, INDIAN_STOCK_MARKET, ON_AND_ON, RMFTSA_LADATA, STOCK, TREASURY, WEATHER_IZMIR],
    REGRESSION: [CONCRETE_DATA, RED_WINE_QUALITY, WHITE_WINE_QUALITY],
    CLASSIFICATION: [AIR_QUALITY, DEBUTANIZER, DIABETES, FOOTBALL_PLAYER_POSITION, GENDER_RECOGNITION],
    LINESTYLE: ['-', '--', ':']
}

DATASET_TYPE_MAP = {
    # MANIFOLD
    SPHERE: MANIFOLD,
    SWISS_ROLL: MANIFOLD,
    TORUS: MANIFOLD,
    # TIME_SERIES
    BOSTON_WEATHER: TIME_SERIES,
    CNN_STOCK_PRED_DJI: TIME_SERIES,
    HUNGARIAN_CHICKENPOX: TIME_SERIES,
    INDIAN_STOCK_MARKET: TIME_SERIES,
    ON_AND_ON: TIME_SERIES,
    RMFTSA_LADATA: TIME_SERIES,
    STOCK: TIME_SERIES,
    TREASURY: TIME_SERIES,
    WEATHER_IZMIR: TIME_SERIES,
    # REGRESSION
    CONCRETE_DATA: REGRESSION,
    RED_WINE_QUALITY: REGRESSION,
    WHITE_WINE_QUALITY: REGRESSION,
    # CLASSIFICATION
    AIR_QUALITY: CLASSIFICATION,
    DEBUTANIZER: CLASSIFICATION,
    DIABETES: CLASSIFICATION,
    FOOTBALL_PLAYER_POSITION: CLASSIFICATION,
    GENDER_RECOGNITION: CLASSIFICATION,
}

### LABELS ###
LABEL: Dict[str, str] = {
    STOCK: 'Stock Dataset',
    RMFTSA_LADATA: 'RMFTSA LA Data',
    CONCRETE_DATA: 'Concrete Data',
    DEBUTANIZER: 'Debutanizer',
    TREASURY: 'Treasury',
    WEATHER_IZMIR: 'Weather Izmir',
    HUNGARIAN_CHICKENPOX: 'Hungarian Chickenpox',
    CNN_STOCK_PRED_DJI: 'CNN Stock Prediction DJI',
    DIABETES: 'Diabetes Dataset',
    INDIAN_STOCK_MARKET: 'Indian Stock Market',
    GENDER_RECOGNITION: 'Gender Recognition by Voice',
    BOSTON_WEATHER: 'Boston Weather Data',
    RED_WINE_QUALITY: 'Red Wine Quality',
    WHITE_WINE_QUALITY: 'White Wine Quality',
    AIR_QUALITY: 'Air Quality and Pollution',
    FOOTBALL_PLAYER_POSITION: 'Football Player Position',
    TORUS: 'Torus (Synthetic)',
    SWISS_ROLL: 'Swiss Roll (Synthetic)',
    SPHERE: 'Sphere (Synthetic)',
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
    VR: 'Vietoris-Rips Complex',
    DTMS: r'Distance To a Measure ($k = \sqrt{n}$)',
    DTMC: r'Distance To a Measure ($k = \lceil m n \rceil,\, m = 0.05$)',
    KD: 'Kernel Distance',
    PD: 'Persistence Diagram',
    PL: 'Persistence Landscape',
    PI: 'Persistence Image',
    WS: 'Wasserstein Distance',
    BN: 'Bottleneck Distance',
    L2PL: r'L$^2$ Distance (Persistence Landscape)',
    L2PI: r'L$^2$ Distance (Persistence Image)',
    RMSE: 'Root Mean Squared Error',
    MAE: 'Mean Absolute Error',
    DATASET: 'Dataset',
    MISSINGNESS_TYPE: 'Type of Missingness',
    MISSING_RATE: 'Missing Rate [%]',
    IMPUTATION_METHOD: 'Imputation Method',
    TDA_METHOD: 'TDA Method',
    DIMENSION: 'Homological Dimension',
    MANIFOLD: 'Synthetic Manifold',
    TIME_SERIES: 'Time Series Analysis',
    REGRESSION: 'Regression',
    CLASSIFICATION: 'Classification'
}

LABEL_SHORT: Dict[str, str] = {
    MCAR: 'MCAR',
    MAR: 'MAR',
    MNAR: 'MNAR',
    DIMENSION: 'Dim'
}