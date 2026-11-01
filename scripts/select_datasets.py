"""
Reproduce the selection of the OpenML datasets (run on 2025-09-29; restored from the former
notebooks/find-datasets.ipynb). Prints the number of datasets left after each step and compares the result
with the datasets in src/constants.py.

    python scripts/select_datasets.py                 # full selection, downloads every candidate once
    python scripts/select_datasets.py --no-load-check # skip the download check (step 4)
"""
import argparse
import os
import sys
import warnings

import openml
from sklearn.datasets import fetch_openml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import log
from src.constants import DATASETS

warnings.filterwarnings('ignore')

# datasets uploaded after the selection are ignored: the manual lists below cover every candidate that existed
# then, and their largest ID is 47038 (OpenML IDs are sequential)
MAX_DID = 47038

# 2. feature and sample bounds
MIN_NUM_FEATURES, MAX_NUM_FEATURES = 8, 50
MIN_SAMPLES, MAX_SAMPLES = 500, 5000

# 5. manual removal, by inspection of each remaining dataset
NON_NUMERIC = [42351, 43127, 43323, 43344, 43352, 43386, 43440, 43442, 43540, 43563, 43581, 43696, 43699,
               43795, 43807, 43817, 43894, 44029, 44203, 44212, 44962, 46381, 46551, 46585, 46587, 46589,
               46597, 46604, 46635, 46731, 46735, 46815, 46883, 46884, 47001, 47012, 47038]
CATEGORICAL = [504, 522, 547, 1028, 4540, 41491, 41553, 41943, 41944, 42188, 42197, 42437, 42438, 42441,
               42444, 42445, 42464, 42636, 43222, 43252, 43338, 43403, 43465, 43582, 43588, 43682, 43701,
               43808, 43826, 44960, 44994, 45033, 45054, 45929, 45950, 46131, 46142, 46144, 46168, 46255,
               46295, 46356, 46416, 46502, 46532, 46555, 46607, 46618, 46761]
DUPLICATES = [
    43448, 46917,                # keep 4353 (Concrete Data)
    46254, 43483, 46572,         # keep 43384 (Diabetes Dataset)
    43406, 42184, 44972,         # keep 43695 (Red Wine Quality)
    45539, 45540, 45536, 45537,  # keep 45538 (Microwaves Sensoring), removed below
    43000,                       # cnn_stock_pred_dji: same matrix as 42999 (Hungarian Chickenpox)
]
SYNTHETIC_WITHOUT_TOPOLOGY = [
    595, 598, 654, 633, 593, 620, 641, 582, 606, 589, 627, 643, 608, 586, 646, 581, 623, 592, 604, 584,  # Friedman
    46264, 46265, 46266, 46269, 46267, 46268, 46270, 46271, 46272, 46275, 46273, 46274,                  # MA-BBOB
    4551, 45538, 46620,
]


def select(load_check=True):
    datasets = openml.datasets.list_datasets(output_format='dataframe')
    datasets = datasets[datasets['did'] <= MAX_DID]
    log(f'{len(datasets)} datasets on OpenML with ID <= {MAX_DID}')

    # 1. initial constraints
    datasets = datasets[(datasets['status'] == 'active') & (datasets['NumberOfMissingValues'] == 0) &
                        (datasets['NumberOfSymbolicFeatures'] == 0)]
    log(f'{len(datasets)} active datasets without missing values and symbolic features')

    # 2. feature and sample bounds
    datasets = datasets[datasets['NumberOfNumericFeatures'].between(MIN_NUM_FEATURES, MAX_NUM_FEATURES) &
                        datasets['NumberOfInstances'].between(MIN_SAMPLES, MAX_SAMPLES)]
    log(f'{len(datasets)} datasets with {MIN_NUM_FEATURES}-{MAX_NUM_FEATURES} numeric features and {MIN_SAMPLES}-{MAX_SAMPLES} instances')

    # 3. newest version per name
    datasets = datasets.sort_values(['name', 'version'], ascending=[True, False]).drop_duplicates('name')
    log(f'{len(datasets)} datasets after keeping the newest version per name')

    # 4. datasets that can be loaded
    if load_check:
        unloadable = []
        for did in datasets['did']:
            try:
                fetch_openml(data_id=int(did), as_frame=True)
            except Exception:
                unloadable.append(did)
        datasets = datasets[~datasets['did'].isin(unloadable)]
        log(f'{len(datasets)} datasets that can be loaded')

    # 5. manual removal
    for label, ids in [('non-numeric features', NON_NUMERIC), ('categorical features', CATEGORICAL),
                       ('duplicates', DUPLICATES), ('synthetic data without topological structure', SYNTHETIC_WITHOUT_TOPOLOGY)]:
        datasets = datasets[~datasets['did'].isin(ids)]
        log(f'{len(datasets)} datasets after removing {label}')
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-load-check', action='store_true', help='skip downloading every candidate')
    datasets = select(load_check=not parser.parse_args().no_load_check)
    print(datasets.sort_values('did')[['did', 'name', 'version', 'NumberOfInstances', 'NumberOfNumericFeatures']].to_string(index=False))
    selected, used = set(datasets['did'].astype(int)), {i for i in DATASETS if i > 0}
    log(f'matches src/constants.py: {selected == used}; missing: {sorted(used - selected)}; '
        f'additional: {sorted(selected - used)}')


if __name__ == '__main__':
    main()
