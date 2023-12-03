# Raincouver Precipitation Prediction
☀️🌤️⛅️🌥️☁️🌦️🌧️⛈️🌩️🌨️❄️

Author: Dan Zhang, Doris (Yun Yi) Cai, Hayley (Yi) Han & Sivakorn (Oak) Chong

## Project Overview 

Our project investigates the prediction of daily precipitation in Vancouver using machine learning methods. Using a dataset spanning from 1990 to 2023, we explored the predictive power of some key environmental and cliamte features such as temperature, wind speed, and evapotranspiration. Our results suggest the best classification model is Support Vector Machine with Radial Basis Function (SVM RBF) model with the hyperparameter C=10.0. The model achieved a notable F1 score of 0.87 on the positive class (precipitation is present) when generalized to the unseen data, suggesting a high accuracy in precipitation prediction. We also explored feature importance, showing ET₀ reference evapotranspiration and the cosine transformation of months as robust predictors. Hyperparameter optimization did not make improvement to our curren model, indicating the potential need for feature engineering or incoportating more features. Our preject presents a reliable model for predicting precipitation with potential practical applications in various fields.

The dataset we used in this project contains daily precipitation information in Vancouver from 1990 to the present (i.e., 6 Nov, 2023). It is sourced from Open-Meteo’s Historical [Weather API](https://doi.org/10.5281/ZENODO.7970649) [1]. Each row in the dataset includes weather measurement statistics in a day. The key measurements in our dataset are month, daily temperature measures, wind speeds, wind direction, shortwave radiation, and ET₀ reference evapotranspiration. Specifically, shortwave radiation represents the sum of solar energy received in a day; ET₀ reference evapotranspiration provides an indication of the atmospheric demand for moisture (i.e., higher relative humidity reduces ET₀ ); and month is also included as a variable since it accounts for the seasonal variations in precipitation [2]. 

## Report

The final report is available [page](https://ubc-mds.github.io/RaincouverPrediction/).

## Dependencies

- Docker is used in this project for software dependencies management. The Docker image for this this project is built up on `quay.io/jupyter/minimal-notebook:2023-11-19`. In the [Dockerfile](https://github.com/UBC-MDS/RaincouverPrediction/blob/main/Dockerfile), it specifies additional packages and dependencies required for this project.

## Usage

#### Setup:

1. Clone the GitHub repository for this project.
   
2. [Install](https://www.docker.com/get-started/) and launch Docker on local computer and keep it run.

#### Analysis:

3. On terminal, navigate to the project local root directry and run the following command to start and run the container. Note: please ensure your localhost port 8888:8888 is not occupied before you run the command above.
   ```
   docker compose up
   ```
   
4. To bring up the web app for Jupyter Notebook in the container, look for the url starting with `http://127.0.0.1:8888/lab?token=` in terminal and copy it to browser.

5. To run the analysis, using the following commands in the terminal in the project root:

```
# download and extract data
python scripts/download_data.py \
        --url="https://archive-api.open-meteo.com/v1/archive" \
        --start-date=1990-01-01 \
        --end-date=2023-11-06\
        --write-to="./data"

# perform eda and save plots
python scripts/eda.py  \
  --data-file=data/van_weather_1990-01-01_2023-11-06.csv \
  --plot-to=results/figures

# data preprocess，split data into train and test sets,
# and save preprocessor
python scripts/drop_split_preprocess.py \
  --data-file=data/van_weather_1990-01-01_2023-11-06.csv \
  --data-to=data/processed  \
  --preprocessor-to=results/models \
  --seed=522

# model selection, model evaluation on test data and save results
python scripts/classification.py \
    --x_train=data/processed/X_train.csv \
    --y_train=data/processed/y_train.csv \
    --x-test=data/processed/X_test.csv \
    --y-test=data/processed/y_test.csv \
    --preprocessor=results/models/precipit_preprocessor.pickle \
    --columns-to-drop=data/processed/columns_to_drop.csv \
    --pipeline-to=results/models \
    --plot-to=results/figures \
    --seed=522

# build HTML report and copy build to docs folder
jupyter-book build reports/milestone3
cp -r reports/milestone3/_build/html/* docs
```


#### Exit container:

6. In terminal, hit `Cntrl` + `C` to stop running the container. Then use the following command to remove the container:
   ```
   docker compose rm -f
   ```

## Developer notes

#### Adding a new dependency
1. Create a new branch and add new dependency to the [Dockerfile](https://github.com/UBC-MDS/RaincouverPrediction/blob/main/Dockerfile) file, re-build the Docker image using the following commmand to ensure it works properly.
```
docker compose up
```

2. Once the updated `Dockerfile` is pushed to GitHub, a new docker image and tag created by the new `Dockerfile` will be synchronized and publish to Docker Hub automatically as the workflow has been set up on Github repository for this project.
 
3. Update the image acorrdingly in the [docker-compose.yml](https://github.com/UBC-MDS/RaincouverPrediction/blob/main/docker-compose.yml) file and ensure the container launches properly

4. Commit the changes on `docker-compose.yml` to the project repo.

5. Send PR to merge changes to 'main' branch.

#### Running the tests
Navigate to the project root directory and use the following command in terminal to test the [functions](https://github.com/UBC-MDS/RaincouverPrediction/tree/main/src) defined in the projects. Tests are stored in [here](https://github.com/UBC-MDS/RaincouverPrediction/tree/main/test).
```
pytest test/*
```

## License

The Raincouver Precipitation Prediction materials are licensed under [MIT License](https://opensource.org/license/mit/). If re-using/re-mixing please provide attribution and link to this webpage.


## Reference

[1] Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. [https://doi.org/10.5281/ZENODO.7970649](https://doi.org/10.5281/ZENODO.7970649)

[2] Pal, Jeremy S., Eric E. Small, and Elfatih AB Eltahir. "Simulation of regional‐scale water and energy budgets: Representation of subgrid cloud and precipitation processes within RegCM." Journal of Geophysical Research: Atmospheres 105.D24 (2000): 29579-29594.
