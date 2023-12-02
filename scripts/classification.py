import click
import os
import altair as alt
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from joblib import dump

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--columns-to-drop', type=str, help="Optional: columns to drop")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(X_train,y_train, preprocessor, columns_to_drop, pipeline_to, plot_to, seed): 
    '''Fits a classifier to the training data, saves pipeline object and produce relevant figures'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    #Read in data and preprocessor
    X_train = pd.read_csv(X_train)
    y_train = pd.read_csv(y_train)

    if columns_to_drop:
        to_drop = pd.read_csv(columns_to_drop).feats_to_drop.tolist()
        X_train = X_train.drop(columns=to_drop)