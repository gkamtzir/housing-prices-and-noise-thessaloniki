from enhance import enhance_properties_with_noise
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import pandas as pd
import matplotlib.pyplot as plt
from evaluate import lime_instance_evaluation, lime_aggregated_feature_evaluation, \
    lime_aggregated_features_evaluation, shap_evaluation
from xgboost import XGBRegressor
from test_cases import filter_based_on_test_case
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np

if __name__ == "__main__":
    # These are the best performing models.

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
    model = lgb.LGBMRegressor(colsample_bytree=0.6609840999909818,
                              learning_rate=0.07216196668844649,
                              max_depth=10,
                              n_estimators=877,
                              num_leaves=120)
    df = filter_based_on_test_case(df, 3, "C")
