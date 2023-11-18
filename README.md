# Raincouver Precipitation Prediction
☀️🌤️⛅️🌥️☁️🌦️🌧️⛈️🌩️🌨️❄️
  - author: Dan Zhang, Doris (Yun Yi) Cai, Hayley (Yi) Han & Sivakorn (Oak) Chong

## Project Overview 

Prediction of daily precipitation is a fundamental aspect of meteorological studies [1]. Accurate precipitation prediction is crucial for agriculture, water resources management, as well as daily activities of people. Specifically, in a geographically and climatically diverse region like Vancouver, predicting precipitation is vital for people to prepare for extreme weather events, reducing hazards and minimizing property damage.

In this project, we aim to predict the occurrence and the amount of daily precipitation in Vancouver using machine learning (ML) classification methods [2]. Specifically, our analysis utilizes a dataset containing daily precipitation information in Vancouver from 1990 to the present (i.e., 6 Nov, 2023). This dataset, sourced from Open-Meteo’s Historical Weather API [3], includes a number of parameters relevant to precipitation prediction. Key parameters include month, daily temperature measures, wind speeds, wind direction, shortwave radiation, and ET₀ reference evapotranspiration. Specifically, shortwave radiation represents the sum of solar energy received in a day; ET₀ reference evapotranspiration provides an indication of the atmospheric demand for moisture (i.e., higher relative humidity reduces ET₀ ); and month is also included as a variable since it accounts for the seasonal variations in precipitation [4]. This project may contributes insights into accurate forecast of the precipitation in Vancouver.

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
- Python and packages listed in [`environment.yml`](environment.yml)

## License

The Raincouver Precipitation Prediction materials are licensed under [MIT License](https://opensource.org/license/mit/). If re-using/re-mixing please provide attribution and link to this webpage.


## Reference

[1] New, Mark, et al. "Precipitation measurements and trends in the twentieth century." International Journal of Climatology: A Journal of the Royal Meteorological Society 21.15 (2001): 1889-1922.

[2] Ortiz-García, E. G., S. Salcedo-Sanz, and C. Casanova-Mateo. "Accurate precipitation prediction with support vector classifiers: A study including novel predictive variables and observational data." Atmospheric research 139 (2014): 128-136.

[3] Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. [https://doi.org/10.5281/ZENODO.7970649](https://doi.org/10.5281/ZENODO.7970649)

[4] Pal, Jeremy S., Eric E. Small, and Elfatih AB Eltahir. "Simulation of regional‐scale water and energy budgets: Representation of subgrid cloud and precipitation processes within RegCM." Journal of Geophysical Research: Atmospheres 105.D24 (2000): 29579-29594.
