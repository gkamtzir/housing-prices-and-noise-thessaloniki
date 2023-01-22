import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import json


def filter_points_in_area_of_interest(row, polygon: Polygon) -> bool:
    """
    Filters the points based on whether they are located
    in the area of interest.
    :param row: The row of the dataframe.
    :param polygon: The polygon.
    :return: True when it's in the area, False otherwise.
    """
    point = Point(row["Longitude"], row["Latitude"])
    return polygon.contains(point) or point.touches(polygon)


def get_polygon(area: str) -> Polygon:
    """
    Get the polygon data for the given area.
    :param area: The key of the area.
    :return: The polygon of the area.
    """
    with open(f"./configuration/areas.json", "r") as f:
        polygon = json.load(f)

    longitudes = list(map(lambda x: x[0], polygon[area]))
    latitudes = list(map(lambda x: x[1], polygon[area]))

    longitudes_latitudes_vector = np.column_stack((longitudes, latitudes))
    return Polygon(longitudes_latitudes_vector)


def filter_based_on_test_case(df: pd.DataFrame, noise: int, area: str = None):
    """
    Filters the given dataframe based on the test case.
    :param df: The dataframe to be filtered.
    :param noise: The noise usage.
    :param area: The area.
    :return: The filtered dataframe.
    """
    if noise == 0:
        df = df.drop(columns=["noise_day", "noise_night"])
    elif noise == 2:
        df["noise"] = (df["noise_day"] + df["noise_night"]) / 2
        df = df.drop(columns=["noise_day", "noise_night"])
    elif noise == 3:
        df = df.drop(columns=["noise_night"])
    elif noise == 4:
        df = df.drop(columns=["noise_day"])

    # Filtering based on the area if needed.
    if area is not None:
        polygon = get_polygon(area)
        mask = df.apply(filter_points_in_area_of_interest, args=(polygon, ), axis=1)
        df = df[mask]

    return df
