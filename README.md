# TDA Imputation Robustness
Empirical study to evaluate "The Robustness of Topological Data Analysis under Imputation-Induced Perturbations"

## Setup Instructions
Follow these steps to set up the project environment:

### 1. Create a Conda Environment
Run the following command to create a new conda environment with Python 3.11:

```bash
conda create --name myenv python=3.11
```

You can replace `myenv` with any other environment name you prefer.

### 2. Activate the Environment
Once the environment is created, activate it using:

```bash
conda activate myenv
```

This ensures that you're working within the isolated environment.

### 3. Install Required Packages
Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

This will install the necessary libraries and packages as listed in the requirements.txt file.

### 4. Additional Notes (Optional)
* If you need to update your environment or add new dependencies, modify the requirements.txt file and rerun the installation command:
* To deactivate the environment when you're done, simply run:

```bash
conda deactivate
```

> [!WARNING]  
> Before running experiments, make sure to set the `WORKERS` variable in `src/constants.py` to your desired number of parallel workers.


## Run Experiments

To run the experiments with **all datasets**, use:

```bash
python main.py
```

To run the experiments with a specific subset of datasets, pass their IDs as arguments:

```bash
python main.py 123
```

Multiple dataset IDs can be provided:

```bash
python main.py 123 321
```

### Datasets

| Dataset ID | String Key                  | Dataset Name                |
|------------|-----------------------------|-----------------------------|
| 223        | stock                       | Stock Dataset               |
| 666        | rmftsa_ladata               | RMFTSA LA Data              |
| 4353       | concrete_data               | Concrete Data               |
| 23516      | debutanizer                 | Debutanizer                 |
| 42367      | treasury                    | Treasury                    |
| 42369      | weather_izmir               | Weather Izmir               |
| 42999      | hungarian_chickenpox        | Hungarian Chickenpox        |
| 43000      | cnn_stock_pred_dji          | CNN Stock Prediction DJI    |
| 43384      | diabetes                    | Diabetes Dataset            |
| 43402      | indian_stock_market         | Indian Stock Market         |
| 43437      | gender_recognition_by_voice | Gender Recognition by Voice |
| 43623      | boston_weather_data         | Boston Weather Data         |
| 43695      | red_wine_quality            | Red Wine Quality            |
| 44971      | white_wine_quality          | White Wine Quality          |
| 46762      | air_quality_and_pollution   | Air Quality and Pollution   |
| 46764      | football_player_position    | Football Player Position    |
| -1         | torus                       | Torus (Synthetic)           |
| -2         | swiss_role                  | Swiss Roll (Synthetic)      |
| -3         | sphere                      | Sphere (Synthetic)          |
| -4         | on_and_on                   | On & On (NCS)               |
