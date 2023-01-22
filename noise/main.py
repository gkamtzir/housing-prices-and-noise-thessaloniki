from utilities import convert_to_wgs, create_dataset, calculate_noise, \
    noise_enhancement
from plot import plot_noise_map
from tessellation import apply_tessellation
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


if __name__ == "__main__":
    # Converting to WGS format
    # convert_to_wgs("./primitive data/kalamaria/kalamaria_day")

    # Creating the dataset
    # create_dataset("./primitive data/kalamaria/kalamaria_night_wgs")

    # Calculating the noise
    noise_enhancement("./generated-data/thessaloniki_day.parquet.gzip")
