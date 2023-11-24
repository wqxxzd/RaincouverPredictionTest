from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


def cross_val_model(preprocessor, models, X_train, y_train, classification_metrics):
    """
    Perform cross-validation for multiple machine learning models.

    Parameters:
    -----------
    preprocessor : Pipeline
        The preprocessing pipeline to be applied to the data.
    models : dict
        A dictionary containing machine learning models, where the keys
        are model names and the values are the corresponding model instances.
    X_train : pandas.DataFrame
        The feature matrix for training.
    y_train : pandas.Series
        The target variable for training.
    classification_metrics : list or str
        The evaluation metric(s) for the cross-validation.
        Can be a single metric or a list of metrics.

    Returns:
    --------
    pandas.DataFrame: 
        A DataFrame summarizing the cross-validation results with mean scores
        or each model.


    Example:
    --------
    >>> models = {'RandomForest': RandomForestClassifier(), 'LogisticRegression': LogisticRegression()}
    >>> preprocess = StandardScaler()
    >>> metrics = ['accuracy', 'precision']
    >>> results = cross_val_model(preprocess, models, X_train, y_train, metrics)
    >>> print(results)
                        RandomForest  LogisticRegression
    fit_time_mean            0.123             0.045
    score_time_mean          0.045             0.021
    test_accuracy_mean       0.835             0.912
    train_accuracy_mean      0.968             0.986
    test_precision_mean      0.812             0.932
    train_precision_mean     0.986             0.978
    ...
    """

    cross_val_results = {}
    for model in models:
        pipe = make_pipeline(preprocessor, models[model])
        cross_val_results[model] = pd.DataFrame(cross_validate(pipe, 
                                       X_train, 
                                       y_train, 
                                       return_train_score=True, 
                                       scoring=classification_metrics)).agg(['mean']).round(3).T
        print("Done with", model)

    results_df = pd.concat(cross_val_results, axis='columns')

    return results_df