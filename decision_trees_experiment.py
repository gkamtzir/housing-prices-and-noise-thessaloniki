import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from experiment import run_experiment, create_run_name
from utilities import read_experiment_parameters
from test_cases import filter_based_on_test_case
from skopt.space import Categorical, Integer
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

    run_name = create_run_name("decision_trees", noise, area)

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
    model = DecisionTreeRegressor()

    # Setting up parameters.
    if mode == "bayesian":
        parameters = {
            "criterion": Categorical(["squared_error", "friedman_mse", "absolute_error", "poisson"]),
            "splitter": Categorical(["best", "random"]),
            "max_depth": Integer(2, 20),
            "min_samples_leaf": Integer(5, 80)
        }
    else:
        parameters = {
            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "splitter": ["best", "random"],
            "max_depth": [2, 3, 5, 7, 10, 15, 20],
            "min_samples_leaf": [5, 8, 10, 15, 40, 80]
        }

    # Run the experiment
    run_experiment(model, parameters, X, y, 100, 0.2, "decision-trees",
                   run_name, experiment_details, metadata, mode)
