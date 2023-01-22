import pandas as pd
import geopy.distance
from noise.tessellation import apply_tessellation


def enhance_properties_with_noise(properties_file_name, mode="day"):
    """
    Enhances the properties dataframe with noise information.
    It stores the enhanced dataframe in the './data/ folder.
    :param properties_file_name: The file where the properties are stored.
    :param mode: Indicates the mode, 'day' or 'night', regarding the noise.
    """
    df_properties = pd.read_parquet(f"./data/{properties_file_name}.parquet.gzip")
    if mode == "day":
        df_noise_thessaloniki = pd.read_parquet("./noise/generated-data/thessaloniki_day_noise.parquet.gzip")
        df_noise_kalamaria = pd.read_parquet("./noise/generated-data/kalamaria_day_noise.parquet.gzip")
        df_noise = pd.concat([df_noise_thessaloniki, df_noise_kalamaria])
    else:
        df_noise_thessaloniki = pd.read_parquet("./noise/generated-data/thessaloniki_night_noise.parquet.gzip")
        df_noise_kalamaria = pd.read_parquet("./noise/generated-data/kalamaria_night_noise.parquet.gzip")
        df_noise = pd.concat([df_noise_thessaloniki, df_noise_kalamaria])

    # Using tessellated version
    df_tessellated = apply_tessellation(df_noise)

    # Enhancing properties with noise data
    df_properties[f"noise_{mode}"] = df_properties.apply(aggregate_closest_noise, args=(df_tessellated, 0.1), axis=1)

    # Dropping properties with no noise values.
    df_properties = df_properties.dropna()

    df_properties.to_parquet(f"./data/{properties_file_name}_enhanced.parquet.gzip", compression="gzip")


def aggregate_closest_noise(row, df_noise, radius=0.1):
    """
    Aggregates the noise that is close to the given row.
    :param row: The dataframe row.
    :param df_noise: The noise dataframe.
    :param radius: The desired radius.
    :return: The aggregated noise.
    """
    df_noise["close"] = df_noise.parallel_apply(calculate_distance, args=(row["Latitude"], row["Longitude"], radius), axis=1)
    if not df_noise.query("close == True").empty:
        return df_noise.query("close == True")["noise"].mean()
    else:
        -1


def calculate_distance(row, latitude, longitude, radius=0.1):
    """
    Checks ifthe distance between the given row (which has
    'Latitude' and 'Longitude' features and the given 'latitude'
    and 'longitude' parameters is less than the given radius.
    :param row: The dataframe row.
    :param latitude: The latitude.
    :param longitude: The longitude.
    :param radius: The radius.
    :return: True if the distane is less than radius. Otherwise, False.
    """
    distance = geopy.distance.geodesic((latitude, longitude), (row["latitude"], row["longitude"]))
    return distance.km < radius
