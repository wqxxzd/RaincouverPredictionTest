import pandas as pd
import pytest
import sys
import os
import numpy as np

from sklearn.tree import DecisionTreeClassifier

# Import the get_vancouver_data function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mean_std_cross_val_scores import mean_std_cross_val_scores

# Test for empty training data
def test_empty_data():
    model = DecisionTreeClassifier()
    X_train_empty = np.array([])
    y_train_empty = np.array([])

    with pytest.raises(ValueError):
        mean_std_cross_val_scores(model, X_train_empty, y_train_empty)

# Test for invalid training data
def test_invalid_data_types():
    model = DecisionTreeClassifier()
    X_train_invalid = "invalid data"
    y_train_invalid = "invalid data"

    with pytest.raises(ValueError):
        mean_std_cross_val_scores(model, X_train_invalid, y_train_invalid)

# Test for kwargs
def test_kwargs():
    X_train = np.random.rand(1000, 20)
    y_train = np.random.randint(0, 2, size=1000)

    model = DecisionTreeClassifier()
    result = mean_std_cross_val_scores(model, X_train, y_train, cv=3, scoring='accuracy')

    assert isinstance(result, pd.Series), "The result should be a pandas Series"
    assert all(isinstance(val, str) for val in result), "All values in the result should be strings"
    assert {'fit_time', 'score_time', 'test_score'} == set(result.index), "The result should contain fit_time, score_time and test_score"