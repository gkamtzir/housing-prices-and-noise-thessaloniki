import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
from math import inf
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import json


def convert_to_wgs(file_name):
    """
    Converts the given `*.tif` file to its WGS format
    so that it is enriched with latitude and longitude
    details.
    :param file_name: The file name.
    """
    dst_crs = 'EPSG:4326'
    with rio.open(f"{file_name}.tif") as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
        with rio.open(f"{file_name}_wgs.tif", 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def create_dataset(file_name):
    """
    Creates the dataset given the enriched image.
    :param file_name: The file name of the image.
    """
    with rio.open(f"{file_name}.tif") as dataset:
        shape = dataset.shape

        non_blacks = 0
        data = {
            "latitude": [],
            "longitude": [],
            "color": []
        }

        # Mapping lat/lon to colors
        for i in range(shape[0]):
            for j in range(shape[1]):
                window = rio.windows.Window(j, i, 1, 1)
                clip = dataset.read(window=window)
                long, lat = dataset.xy(i, j)
                red = clip[0][0][0]
                green = clip[1][0][0]
                blue = clip[2][0][0]
                if red != 0 or green != 0 or blue != 0:
                    non_blacks += 1
                    data["latitude"].append(lat)
                    data["longitude"].append(long)
                    data["color"].append((red, green, blue))

        # Creating and storing the dataset
        df = pd.DataFrame(data)

        # Reducing memory size by 20%
        df["red"] = df["color"].map(lambda x: x[0]).astype("uint8")
        df["green"] = df["color"].map(lambda x: x[1]).astype("uint8")
        df["blue"] = df["color"].map(lambda x: x[2]).astype("uint8")
        df = df.drop(columns="color")
        df.to_parquet(f"{file_name}.parquet.gzip", compression="gzip")


def calculate_noise(row, mode="thessaloniki"):
    """
    Calculates the noise of the given row. The row contains the color
    in RGB format. The function converts it to LAB format and then
    computes the difference using DeltaE to map it to the corresponding
    color range.
    :param row: The row data.
    :param mode: The coloring mode ("thessaloniki" and "kalamaria")
    :return: The noise value.
    """
    minimum = inf
    limit = 20
    min_mapping = None
    with open(f"configurations/{mode}_color_configuration.json", "r") as f:
        mapping_data = json.load(f)
    for mapping in mapping_data:
        color1_rgb = sRGBColor(mapping["color"][0], mapping["color"][1], mapping["color"][2])
        color2_rgb = sRGBColor(row["red"], row["green"], row["blue"])

        color1_lab = convert_color(color1_rgb, LabColor)
        color2_lab = convert_color(color2_rgb, LabColor)

        delta_e = delta_e_cie2000(color1_lab, color2_lab)

        if delta_e <= limit and delta_e < minimum:
            minimum = delta_e
            min_mapping = mapping

    if min_mapping is not None:
        return min_mapping["value"]
    else:
        return -1


def noise_enhancement(file_name, mode="thessaloniki"):
    """
    Enhances the dataframe with the given name with a noise column,
    based on the 'calculate_noise' function. The result is saved
    in a new dataframe.
    :param file_name: The filename where the dataframe is stored.
    :param mode: (Optional) The coloring mode. Currently, we support
    colors for Thessaloniki and Kalamaria only.
    """
    df = pd.read_parquet(file_name)
    df["noise"] = df.parallel_apply(calculate_noise, args=(mode,), axis=1)
    df = df.drop(df[df["noise"] == -1].index)
    df.to_parquet("noise.parquet.gzip", compression="gzip")
