import pandas as pd
from xgboost import XGBRegressor
from experiment import run_experiment, create_run_name
from utilities import read_experiment_parameters
from test_cases import filter_based_on_test_case
from skopt.space import Real, Integer
import json


if __name__ == "__main__":
    df = pd.read_parquet("./data/properties_noise_100.parquet.gzip")

    # Reading experiment parameters.
    experiment_id, experiment_name, experiment_tags, noise, mode, area = read_experiment_parameters()

    print(f"Running with noise={noise} and area={area}")

    df = filter_based_on_test_case(df, noise, area)
    print(f"Total number of rows: {df.shape}")

    y = df["Price"]
    X = df.drop("Price", axis=1)

    run_name = create_run_name("xgboost", noise, area)

    metadata = {
        "noise": noise,
        "mode": mode,
        "area": area
    }

    experiment_details = {
        "id": experiment_id,
        "name": experiment_name,
        "artifact": "./data/properties_noise_100.parquet.gzip",
        "tags": json.loads(experiment_tags) if experiment_tags is not None else None
    }

    # Initializing model.
    model = XGBRegressor()

    # Setting up parameters.
    if mode == "bayesian":
        parameters = {
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "n_estimators": Integer(100, 1000),
            "colsample_bytree": Real(0.3, 0.7, prior="log-uniform")
        }
    else:
        parameters = {
            "max_depth": [3],
            # "max_depth": [3, 4, 6, 5, 10],
            # "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            # "n_estimators": [100, 300, 500, 700, 900, 1000],
            # "colsample_bytree": [0.3, 0.5, 0.7]
        }

    # Run the experiment
    run_experiment(model, parameters, X, y, 100, 0.2, "xgboost",
                   run_name, experiment_details, metadata, mode)
