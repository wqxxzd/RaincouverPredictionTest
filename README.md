# Raincouver Precipitation Prediction
☀️🌤️⛅️🌥️☁️🌦️🌧️⛈️🌩️🌨️❄️
  - author: Dan Zhang, Doris (Yun Yi) Cai, Hayley (Yi) Han & Sivakorn (Oak) Chong

## Project Overview 

Our project is to build a classification model to predict if there's precipitation in a day (True or False) and a regression model to predict the amount of precipitation, based on features of temperature, wind speed, direction, shortwave radiation and evapotranspiration. The best classification model in our training and testing process is SVC-RBF with hyperparameter C=10.0. It yields the best test score of 0.8625 and f1-score of 0.87 when generalizes to the unseen data. This is a pretty high accuracy to predict whether there's rain on a particular day. The best regression model trained with the same features to predict the amount of precipitaiton is SVR with gamma=0.1 and C=1000. It produces the best score on the unseen test data of 0.6993. The accuracy is adequate. More study could be done to improve the regression model.

The dataset we used in this project contains daily precipitation information in Vancouver from 1990 to the present (i.e., 6 Nov, 2023). It is sourced from Open-Meteo’s Historical [Weather API](https://doi.org/10.5281/ZENODO.7970649) [1]. Each row in the dataset includes weather measurement statistics in a day. The key measurements in our dataset are month, daily temperature measures, wind speeds, wind direction, shortwave radiation, and ET₀ reference evapotranspiration. Specifically, shortwave radiation represents the sum of solar energy received in a day; ET₀ reference evapotranspiration provides an indication of the atmospheric demand for moisture (i.e., higher relative humidity reduces ET₀ ); and month is also included as a variable since it accounts for the seasonal variations in precipitation [2]. 

## Report

The final report will be available when the report [page]() is created.

## Usage

Before the first time running the project, run the following from the root of this repository:

``` bash
conda env create -f environment.yaml
```

To run the analysis, run the following from the root of this repository:

``` bash
conda activate raincouver_prediction_env
jupyter lab 
```

Open `src/weather_forecast.ipynb` in Jupyter Lab
and under the "Kernel" menu click "Restart Kernel and Run All Cells...".

## Dependencies

- `conda` (version 23.7.4 or higher)
- `nb_conda_kernels` (version 2.3.1 or higher)
- Python and packages listed in [`environment.yaml`](environment.yaml)

## License

The Raincouver Precipitation Prediction materials are licensed under [MIT License](https://opensource.org/license/mit/). If re-using/re-mixing please provide attribution and link to this webpage.


## Reference

[1] Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. [https://doi.org/10.5281/ZENODO.7970649](https://doi.org/10.5281/ZENODO.7970649)

[2] Pal, Jeremy S., Eric E. Small, and Elfatih AB Eltahir. "Simulation of regional‐scale water and energy budgets: Representation of subgrid cloud and precipitation processes within RegCM." Journal of Geophysical Research: Atmospheres 105.D24 (2000): 29579-29594.
