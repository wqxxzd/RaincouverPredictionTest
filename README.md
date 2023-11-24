# Raincouver Precipitation Prediction
☀️🌤️⛅️🌥️☁️🌦️🌧️⛈️🌩️🌨️❄️
Author: Dan Zhang, Doris (Yun Yi) Cai, Hayley (Yi) Han & Sivakorn (Oak) Chong

## Project Overview 

Our project is to build a classification model to predict if there's precipitation in a day (True or False) and a regression model to predict the amount of precipitation, based on features of temperature, wind speed, direction, shortwave radiation and evapotranspiration. The best classification model in our training and testing process is SVC-RBF with hyperparameter C=10.0. It yields the best test score of 0.8625 and f1-score of 0.87 when generalizes to the unseen data. This is a pretty high accuracy to predict whether there's rain on a particular day. The best regression model trained with the same features to predict the amount of precipitaiton is SVR with gamma=0.1 and C=1000. It produces the best score on the unseen test data of 0.6993. The accuracy is adequate. More study could be done to improve the regression model.

The dataset we used in this project contains daily precipitation information in Vancouver from 1990 to the present (i.e., 6 Nov, 2023). It is sourced from Open-Meteo’s Historical [Weather API](https://doi.org/10.5281/ZENODO.7970649) [1]. Each row in the dataset includes weather measurement statistics in a day. The key measurements in our dataset are month, daily temperature measures, wind speeds, wind direction, shortwave radiation, and ET₀ reference evapotranspiration. Specifically, shortwave radiation represents the sum of solar energy received in a day; ET₀ reference evapotranspiration provides an indication of the atmospheric demand for moisture (i.e., higher relative humidity reduces ET₀ ); and month is also included as a variable since it accounts for the seasonal variations in precipitation [2]. 

## Report

The final report is available [page](https://ubc-mds.github.io/RaincouverPrediction/).

## Dependencies

- Docker is used in this project for software dependencies management. The Docker [image]() for this this project is built up on `quay.io/jupyter/minimal-notebook:2023-11-19`. In the [Dockerfile](https://github.com/UBC-MDS/RaincouverPrediction/blob/main/Dockerfile), it specifies additional packages and dependencies required for this project.

## Usage

#### Setup:

1. Clone the GitHub repo using the following command:
   ```
   git clone
   ```
   
2. [Install](https://www.docker.com/get-started/) and launch Docker on local computer and keep it run.

#### Analysis:

3. On terminal, navigate to the project local root directry and run the following command to start and run the container:
   ```
   docker compose up
   ```
   Note: please ensure your localhost port 8888:8888 is not occupied before you run the command above.
   
5. To bring up the web app for Jupyter Notebook in the container, look for the url starting with `http://127.0.0.1:8888/lab?token=` in terminal and copy it to browser.

6. to run the analysis, navigate to and open up the `notebooks/milestone2/weather_forecast.ipynb` in the Jupyter Notebook web app, click "Restart Kernel and Run All Cells..." under the "Kernel" menu.

#### Exit container:

7. In terminal, hit `Cntrl` + `C` to stop running the container. Then use the following command to remove the container:
   ```
   docker compose rm -f
   ```

## License

The Raincouver Precipitation Prediction materials are licensed under [MIT License](https://opensource.org/license/mit/). If re-using/re-mixing please provide attribution and link to this webpage.

## Developer notes

#### Adding a new dependency
1. Create a new branch and add new dependency to the [Dockerfile](https://github.com/UBC-MDS/RaincouverPrediction/blob/main/Dockerfile) file, re-build the Docker image using
```
docker compose up
```
to ensure it works properly.
   
4. Once the updated `Dockerfile` is pushed to GitHub, a new docker image and tag created by the new `Dockerfile` will be synchronized and publish to Docker Hub automatically as the workflow has been set up on Github repository for this project.
 
5. Update the image acorrdingly in the [docker-compose.yml](https://github.com/UBC-MDS/RaincouverPrediction/blob/main/docker-compose.yml) file and ensure the container launches properly

6. Commit the changes on `docker-compose.yml` to the project repo.

7. Send PR to merge changes to 'main' branch.

#### Running the tests
Navigate to the project root directory and use the following command in terminal to test the [functions](https://github.com/UBC-MDS/RaincouverPrediction/tree/main/src) defined in the projects:
```
pytest tests/<function script files>
```
Tests are stored in [here](https://github.com/UBC-MDS/RaincouverPrediction/tree/main/tes).

## Reference

[1] Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. [https://doi.org/10.5281/ZENODO.7970649](https://doi.org/10.5281/ZENODO.7970649)

[2] Pal, Jeremy S., Eric E. Small, and Elfatih AB Eltahir. "Simulation of regional‐scale water and energy budgets: Representation of subgrid cloud and precipitation processes within RegCM." Journal of Geophysical Research: Atmospheres 105.D24 (2000): 29579-29594.
