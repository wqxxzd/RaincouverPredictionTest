import unittest
import pandas as pd
import pytest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import cross_val_model 


# Create a sample dataset for testing
X, y = make_classification(n_samples=100, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dummy preprocessing pipeline
preprocessor = StandardScaler()


# Models for testing
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "LogisticRegression": LogisticRegression()
}

# Classification metrics for testing
classification_metrics = ['accuracy', 'precision']

#Test if the function is returnning a pandas DataFrame
def test_return_type():
    results = cross_val_model(preprocessor, models, X_train, y_train, classification_metrics)
    assert isinstance(results, pd.DataFrame), "Function should return a pandas DataFrame"

#Test if the DataFrame contains the mean scores for all models passed in
def test_different_models():
    test_models = {"KNeighbors": KNeighborsClassifier(), "SVC": SVC()}
    results = cross_val_model(preprocessor, test_models, X_train, y_train, classification_metrics)
    assert all(model in results.columns for model in test_models), "Results should contain all tested models"

#Test if the correct scoring metrics are return as expected
def test_with_different_metrics():
    test_metrics = ['recall', 'f1']
    results = cross_val_model(preprocessor, models, X_train, y_train, test_metrics)
    expected_scores = [f"test_{metric}" for metric in test_metrics]
    assert all(index in results.index for index in expected_scores), "Results should contain the correct metrics"

#Test if scoring metrics passed into the function is a valid stirng or a valid list
def test_with_invalid_input():
    with pytest.raises(ValueError):
        cross_val_model(preprocessor, models, X_train, y_train, "invalid_metric")

#Test if output data type is numeric
def test_output_format():
    results = cross_val_model(preprocessor, models, X_train, y_train, classification_metrics)
    assert all(isinstance(value, (float, np.number)) for value in results.values.flatten()), "All values in the results should be numeric"
