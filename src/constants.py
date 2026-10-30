from typing import Dict

### OpenML Data ###
STOCK_DATASET_ID = 223
RMFTSA_LADATA_DATASET_ID = 666
CONCRETE_DATA_DATASET_ID = 4353
DEBUTANIZER_DATASET_ID = 23516
TREASURY_DATASET_ID = 42367
WEATHER_IZMIR_DATASET_ID = 42369
HUNGARIAN_CHICKENPOX_DATASET_ID = 42999
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

DATASETS = {
    STOCK_DATASET_ID: STOCK,
    RMFTSA_LADATA_DATASET_ID: RMFTSA_LADATA,
    CONCRETE_DATA_DATASET_ID: CONCRETE_DATA,
    DEBUTANIZER_DATASET_ID: DEBUTANIZER,
    TREASURY_DATASET_ID: TREASURY,
    WEATHER_IZMIR_DATASET_ID: WEATHER_IZMIR,
    HUNGARIAN_CHICKENPOX_DATASET_ID: HUNGARIAN_CHICKENPOX,
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
    -3: SPHERE
}

### EXPERIMENTS ###
SEEDS = list(range(1, 10))
MCAR = 'missing_completely_at_random'
MAR = 'missing_at_random'
MNAR = 'missing_not_at_random'
CONSTANT = 'constant_imputation'
MEAN = 'mean_imputation'
MEDIAN = 'median_imputation'
KNN = 'k_nearest_neighbors_imputation'
RF = 'random_forest_imputation'
MICE = 'mice_imputation'
GAIN = 'gain_imputation'
TABCSDI = 'tabcsdi_imputation'
OTIMPUTE = 'otimpute_imputation'
COMPLETE_CASE = 'complete_case_analysis'
RANDOM_SUBSAMPLE = 'random_subsample'
SHUFFLED_GROUND_TRUTH = 'shuffled_ground_truth'
VR = 'vietoris_rips'
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

### PIPELINE ###
WORKERS = 30
N_JOBS = 1
ESTIMATORS = 30
DEPTH = 15
ITERATIONS = 5
GAIN_ITERATIONS = 10000
TABCSDI_EPOCHS = 100
OT_ITERATIONS = 500
DIMENSIONS = [0, 1, 2]
EUCLIDEAN = 2
DETERMINISTIC = 'deterministic'
MST_SCALE_FACTOR = 2.5
MIN_BASELINE_POINTS = 10
SUBSAMPLE_FRACTION = 0.8

### IDENTIFIER ###
DATASET = 'dataset'
MISSINGNESS_TYPE = 'missingness_type'
MISSING_RATE = 'missing_rate'
IMPUTATION_METHOD = 'imputation_method'
BASELINE_METHOD = 'baseline_method'
TDA_METHOD = 'tda_method'
DIMENSION = 'dimension'
N_SEEDS = 'n_seeds'

### COLLECTIONS ###
COLLECTIONS = {
    DATA: [AIR_QUALITY, BOSTON_WEATHER, CONCRETE_DATA, DEBUTANIZER,
           DIABETES, FOOTBALL_PLAYER_POSITION, GENDER_RECOGNITION, HUNGARIAN_CHICKENPOX,
           INDIAN_STOCK_MARKET, RED_WINE_QUALITY, RMFTSA_LADATA, SPHERE,
           STOCK, SWISS_ROLL, TORUS, TREASURY, WEATHER_IZMIR, WHITE_WINE_QUALITY],
    MISSINGNESS_TYPE: [MCAR, MAR, MNAR],
    MISSING_RATE: [5, 10, 25],
    IMPUTATION_METHOD: [CONSTANT, MEAN, MEDIAN, KNN, RF, MICE, GAIN, TABCSDI, OTIMPUTE],
    BASELINE_METHOD: [COMPLETE_CASE, RANDOM_SUBSAMPLE, SHUFFLED_GROUND_TRUTH],
    TDA_METHOD: [VR, DTMC, KD],
    MANIFOLD: [SPHERE, SWISS_ROLL, TORUS],
    TIME_SERIES: [BOSTON_WEATHER, HUNGARIAN_CHICKENPOX, INDIAN_STOCK_MARKET, RMFTSA_LADATA, STOCK, TREASURY, WEATHER_IZMIR],
    REGRESSION: [CONCRETE_DATA, RED_WINE_QUALITY, WHITE_WINE_QUALITY],
    CLASSIFICATION: [AIR_QUALITY, DEBUTANIZER, DIABETES, FOOTBALL_PLAYER_POSITION, GENDER_RECOGNITION]
}

DATASET_TYPE_MAP = {
    # MANIFOLD
    SPHERE: MANIFOLD,
    SWISS_ROLL: MANIFOLD,
    TORUS: MANIFOLD,
    # TIME_SERIES
    BOSTON_WEATHER: TIME_SERIES,
    HUNGARIAN_CHICKENPOX: TIME_SERIES,
    INDIAN_STOCK_MARKET: TIME_SERIES,
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
    STOCK: 'Stock',
    RMFTSA_LADATA: 'RMFTSA LA Data',
    CONCRETE_DATA: 'Concrete Data',
    DEBUTANIZER: 'Debutanizer',
    TREASURY: 'Treasury',
    WEATHER_IZMIR: 'Weather Izmir',
    HUNGARIAN_CHICKENPOX: 'Hungarian Chickenpox',
    DIABETES: 'Diabetes',
    INDIAN_STOCK_MARKET: 'Indian Stock Market',
    GENDER_RECOGNITION: 'Gender Recognition by Voice',
    BOSTON_WEATHER: 'Boston Weather Data',
    RED_WINE_QUALITY: 'Red Wine Quality',
    WHITE_WINE_QUALITY: 'White Wine Quality',
    AIR_QUALITY: 'Air Quality and Pollution',
    FOOTBALL_PLAYER_POSITION: 'Football Player Position',
    TORUS: 'Torus',
    SWISS_ROLL: 'Swiss Roll',
    SPHERE: 'Sphere',
    MCAR: 'Missing Completely at Random',
    MAR: 'Missing at Random',
    MNAR: 'Missing Not at Random',
    CONSTANT: 'Constant',
    MEAN: 'Mean',
    MEDIAN: 'Median',
    KNN: 'k-Nearest Neighbors',
    RF: 'Random Forest',
    MICE: 'MICE',
    GAIN: 'GAIN',
    TABCSDI: 'TabCSDI',
    OTIMPUTE: 'OTImpute',
    COMPLETE_CASE: 'Complete-Case Analysis',
    RANDOM_SUBSAMPLE: 'Random Subsample',
    SHUFFLED_GROUND_TRUTH: 'Shuffled Ground Truth',
    VR: 'Vietoris-Rips Filtration',
    DTMC: r'Distance To a Measure ($k = \lceil m n \rceil,\, m = 0.05$)',
    KD: 'Kernel Distance',
    PD: 'Persistence Diagram',
    PL: 'Persistence Landscape',
    PI: 'Persistence Image',
    WS: 'Wasserstein Distance',
    BN: 'Bottleneck Distance',
    L2PL: r'L$^2$ Distance (Landscapes)',
    L2PI: r'L$^2$ Distance (Images)',
    0: r'H$_0$',
    1: r'H$_1$',
    2: r'H$_2$',
    RMSE: 'Root Mean Square Error',
    MAE: 'Mean Absolute Error',
    DATASET: 'Dataset',
    MISSINGNESS_TYPE: 'Type of Missingness',
    MISSING_RATE: 'Missing Rate [%]',
    IMPUTATION_METHOD: 'Imputation Method',
    TDA_METHOD: 'TDA Method',
    DIMENSION: 'Homological Dimension',
    MANIFOLD: 'Synthetic Manifold',
    TIME_SERIES: 'Time Series',
    REGRESSION: 'Regression',
    CLASSIFICATION: 'Classification',
    BASELINE_METHOD: 'Baseline Method',
}