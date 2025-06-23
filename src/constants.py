### OpenML Data ###
MIN_NUM_FEATURES = 8
MAX_NUM_FEATURES = 25
MIN_SAMPLES = 500
MAX_SAMPLES = 1500
DATASET_KEYS = ['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']
STOCK_DATASET_ID = 223
RMFTSA_LADATA_DATASET_ID = 666
CONCRETE_DATA_ID = 4353
TREASURY_DATASET_ID = 42367
WEATHER_IZMIR_DATASET_ID = 42369
HUNGARIAN_CHICKENPOX_DATASET_ID = 42999
CNN_STOCK_PRED_DJI_DATASET_ID = 43000
DIABETES_DATASET_ID = 43384

# Torus
TORUS_SAMPLES = 1000
TORUS_AMBIENT = 20
TORUS_SEED = 42

# On an On (NCS)
AUDIO_PATH = 'data/on_and_on_ncs.wav'
HOP_LENGTH = 512
WINDOW_DURATION = 0.5 
STEP_DURATION = 0.2
CHROMA_LABELS = ['C', 'Csharp', 'D', 'Dsharp', 'E', 'F', 'Fsharp', 'G', 'Gsharp', 'A', 'Asharp', 'B']

### DATA ###
DATA = 'data'
TARGET = 'target'
FUNCTION = 'function'
CONCRETE_DATA = 'concrete_data'
DIABETES = 'diabetes'
STOCK = 'stock'
RMFTSA_LADATA = 'rmftsa_ladata'
TREASURY = 'treasury'
WEATHER_IZMIR = 'weather_izmir'
HUNGARIAN_CHICKENPOX = 'hungarian-chickenpox'
CNN_STOCK_PRED_DJI = 'cnn-stock-pred-dji'
TORUS = 'torus'
ON_AND_ON = 'on_and_on'

### PIPELINE ###
WORKERS = 8
N_JOBS = 1
ESTIMATORS = 30
DEPTH = 15
ITERATIONS = 5
MEDIAN = 0.5
DIMENSIONS = [0, 1, 2]
EUCLIDEAN = 2
DETERMINISTIC = 'deterministic'
PD = 'persistence_diagram'
PL = 'persistence_landscape'
PI = 'persistence_image'
COMPARISONS = [PD, PL, PI]

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
WS = 'wasserstein_distance'
BN = 'bottleneck_distance'
L2PL = 'persistence_landscape_l2_distance'
L2PI = 'persistence_image_l2_distance'
