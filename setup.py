from setuptools import setup, find_packages


setup(
    name="HousingPricePredictionDataGeneration",
    version="1.0.0",
    url="https://github.com/gkamtzir/housing-price-prediction-data-generation",
    author="Georgios Kamtziridis, Grigorios Tsoumakas",
    author_email="georgiok@csd.auth.gr, greg@csd.auth.gr",
    description="Generating and collecting required data for the Housing Price Prediction diploma thesis",
    packages=find_packages(),
    install_requires=["pandas", "numpy == 1.21.0", "plotly", "GDAL", "rasterio", "colormath", "pandarallel",
                      "matplotlib", "shapely", "category-encoders", "geopy", "scikit-learn", "mlflow",
                      "xgboost", "scikit-optimize", "lightgbm", "lime", "shap"],
)