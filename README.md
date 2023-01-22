# Does Noise Affect Housing Prices? A Case Study in the Urban Area of Thessaloniki

## Introduction
This repository consists of the coding infrastructure required to run the experiments mentioned
in the corresponding paper. Additionally, the datasets used are provided via GDrive
links further below.

## Installation Requirements
First, make sure you have installed all the needed packages.
To do so, just run:
```bash
pip install .
```
**Important note**: If you are using a Windows machine, you
will have to manually download both [GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) and [rasterio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio).
Visit the two websites and download the versions that match your
Python version. Then run the following:
```bash
pip install -U pip
pip install /path/to/gdal/wheel/*.whl
pip install /path/to/rasterio/wheel/*.whl
```

## MLflow
The entire machine learning development is tracked by [MLflow](https://github.com/mlflow/mlflow/).
The parameters and the metrics of each experiment are logged in the `mlruns` folder and
can be viewed in via the MLflow UI. To do so, run:
```bash
mlflow ui
```

## Noise
In the `noise` folder there are primitive data and
scripts needed to reconstruct noise data for the
municipalities of Thessaloniki, Neapoli and Kalamaria,
in the urban area of Thessaloniki. The primitive data
are basically high resolution heatmaps with noise
information. These heatmaps were enriched with latitude
and longitude details through the process of Georeferencing.
So, for each heatmap there is the corresponding `*.tif` file,
which spatial information.

The reconstructed noise datasets are located in the `noise/generated-data` folder.

| File                                        | Description                                                      |
|---------------------------------------------|------------------------------------------------------------------|
| thessaloniki_day_noise.parquet.gzip         | Day noise dataset for Thessaloniki and Neapoli (overall noise)   |
| thessaloniki_night_noise.parquet.gzip       | Night noise dataset for Thessaloniki and Neapoli (overall noise) |
| kalamaria_day_noise.parquet.gzip            | Day noise dataset for Kalamaria (overall noise)                  |
| kalamaria_night_noise.parquet.gzip          | Night noise dataset for Kalamaria (overall noise)                |
| kalamaria_aviation_day_noise.parquet.gzip   | Aviation day noise dataset for Kalamaria                         |
| kalamaria_aviation_night_noise.parquet.gzip | Aviation night noise dataset for Kalamaria                       |

**All files, both generated and primitive data, can be found in [this](https://drive.google.com/drive/folders/142-YkH6WTpKnRS0YuA8rglowneBqvTML?usp=sharing) GDrive folder.**

## Openhouse Properties
The final version of the Openhouse properties must be moved from GDrive's `properties`
folder to the `data` folder of this repository. There are 2 version each with a different radius (50 and 100 meters).

## Experiments
Each model has its own experiment file which can be executed via command line by providing
the required arguments. These are:
- `--noise {noise_id}` 0 -> no noise, 1 -> both day and night noise, 2 -> average of day and night noise, 3 -> only day noise, 4 -> only night noise
- `--mode {hyperparameter_tuning_id}` `'grid_search'` -> Grid Search, `'bayesian'` -> Baysian, else -> Random
- `--area {area_id}` 'A' -> center, 'B' -> Triandria, Toumpa, Harilaou, 'C' -> Kalamaria
- `--experiment-name {name}` The name of the experiment to be created
- `--experiment-tags {tags}` A dictionary containing any extra information
- `--experiment-id {id}` The id of an already existing experiment

### Examples
To train an XGBoost with Bayesian optimization in Kalamaria using both day and night noises run:
```commandline
py .\decision_trees_experiment.py --noise 1 --mode "bayesian" --area "C"  --experiment-name "Test"
```

## Evaluation
The implementation of the evaluation techniques can be found in the `evaluate.py` file.

