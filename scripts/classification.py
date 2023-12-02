import click
import os
import altair as alt
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, make_scorer
from joblib import dump
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import cross_val_model

@click.command()
@click.option('--x_train', type=str, help="Path to X training data")
@click.option('--y_train', type=str, help="Path to y training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object", default=None)
@click.option('--columns-to-drop', type=str, help="Optional: columns to drop")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
# @click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(x_train, y_train, preprocessor, columns_to_drop,
         pipeline_to, seed): 
    '''Fits a classifier to the training data, saves pipeline object and produce relevant figures'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    #Read in data and preprocessor
    X_train = pd.read_csv(x_train)
    y_train = pd.read_csv(y_train)
    y_col_name = y_train.columns.tolist()[0]
    
    y_train_class = y_train[y_col_name]
    preprocess = pickle.load(open(preprocessor, "rb"))
    
    if columns_to_drop:
        to_drop = pd.read_csv(columns_to_drop).feats_to_drop.tolist()
        X_train = X_train.drop(columns=to_drop)

    click.echo(f'Running model training for {X_train.columns.tolist()}')
    
    #Model type selection
    models = {
    "Decision Tree": DecisionTreeClassifier(random_state=522),
    "KNN": KNeighborsClassifier(),
    "RBF SVM": SVC(random_state=522),
    "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="ovr", random_state=522),
    }

    classification_metrics = ["accuracy", "precision", "recall", "f1"]
    results_df = cross_val_model(preprocess, models, X_train, y_train_class, classification_metrics)

    #Choose the model based on the best F1 score. 
    result_dict = results_df.loc['test_f1', :].to_dict()
    
    #Get model name with max_score
    max_score = max(result_dict, key=result_dict.get)
    model_name = max_score[0]

    #Echo out if the model chosen is not expected as SVM RBF
    if model_name == 'RBF SVM':
        click.echo('Proceed with hyperparameter optimization for RBF SVM model')
    else:
        click.echo('Best model is not expected SVM RBF. Other model parameters required for optimization, exitting the program')
        sys.exit(0)
        
    #Hyperparameter optimization
    param_grid = {"svc__C": 10.0**np.arange(-3,3)}
    svc_pipe = make_pipeline(preprocess, models[model_name])
    grid_search = GridSearchCV(svc_pipe,param_grid=param_grid,n_jobs=-1,return_train_score=True)
    grid_search.fit(X_train, y_train_class)
    C_best_value = grid_search.best_params_['svc__C']
    opt_pipe = make_pipeline(preprocess, SVC(C=C_best_value, random_state = 522))
    
    opt_pipe.fit(X_train, y_train_class)

    #Saving optimum pipe as a pickle file for testing
    with open(os.path.join(pipeline_to, "optimum_cls_svm_pipeline.pickle"), 'wb') as f:
        pickle.dump(opt_pipe, f)

    click.echo(f'Optimized model has been parked at {os.path.join(pipeline_to, "optimum_cls_svm_pipeline.pickle")}')

    #Print out feature importance
    

if __name__ == '__main__':
    main()