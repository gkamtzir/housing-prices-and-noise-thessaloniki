import plotly.express as px


def plot_noise_map(df):
    """
    Plots the noise map for the given dataframe. The
    dataframe must include 'latitude' and 'longitude' columns,
    as well as the noise column.
    :param df: The dataframe to be plotted.
    """
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='noise',
                            mapbox_style="stamen-terrain")
    fig.show()
