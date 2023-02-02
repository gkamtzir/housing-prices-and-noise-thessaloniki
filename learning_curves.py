from xgboost import XGBRegressor
import lightgbm as lgb
from test_cases import filter_based_on_test_case
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

if __name__ == "__main__":
    # These are the best performing models.

    # XGBoost (r=50, noise=1, area="A")
    # df = pd.read_parquet("./data/properties_noise_50.parquet.gzip")
    # model = XGBRegressor(colsample_bytree=0.6754824399235599,
    #                      learning_rate=0.041788775351237435,
    #                      max_depth=5,
    #                      n_estimators=1000)
    # df = filter_based_on_test_case(df, 1, "A")

    # XGBoost (r=100, noise=1, area="B")
    # df = pd.read_parquet("./data/properties_noise_100.parquet.gzip")
    # model = XGBRegressor(colsample_bytree=0.6812720467459926,
    #                      learning_rate=0.0436683325289631,
    #                      max_depth=7,
    #                      n_estimators=279)
    # df = filter_based_on_test_case(df, 1, "B")

    # LGBM (r=50, noise=3, area="C")
    df = pd.read_parquet("./data/properties_noise_50.parquet.gzip")
    model = lgb.LGBMRegressor(colsample_bytree=	0.6609840999909818,
                              learning_rate=0.07216196668844649,
                              max_depth=10,
                              n_estimators=877,
                              num_leaves=120)
    df = filter_based_on_test_case(df, 3, "C")

    print(df.shape)

    y = df["Price"]
    X = df.drop("Price", axis=1)

    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 18),
        "cv": ShuffleSplit(n_splits=5, test_size=0.2, random_state=0),
        "score_type": "test",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Negative MAE",
        "scoring": "neg_mean_absolute_error"
    }

    LearningCurveDisplay.from_estimator(model, **common_params)
    plt.show()
