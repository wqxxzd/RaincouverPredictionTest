import pandas as pd
import pytest
import sys
import os

# Import the count_classes function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.get_api import get_vancouver_data

# Test for correct return type with 10 date range for sampling
start_date = "2010-01-01"
end_date = "2010-01-10"

# Test for correct return type
def test_get_vancouver_data_returns_dataframe():
    result = get_vancouver_data(start_date, end_date)
    assert isinstance(result, pd.DataFrame), "get_vancouver_data should return a pandas data frame"

# Test for correct number of rows
def test_get_vancouver_data_number_of_rows():
    result = get_vancouver_data(start_date, end_date)
    assert result.shape[0] == 10, "Returned wrong number of rows"

# Test for correct number of columns
def test_get_vancouver_data_number_of_cols():
    result = get_vancouver_data(start_date, end_date)
    assert result.shape[1] == 18, "Returned wrong number of cols"

# Test for correct the type of index
def test_get_vancouver_data_date_index():
    result = get_vancouver_data(start_date, end_date)
    assert isinstance(result.index, pd.DatetimeIndex), "Did not return DatetimeIndex"

# Test if the date in between the start and end dates is correctly displayed in datetimeindex
def test_get_vancouver_data_date_mid_index():
    result = get_vancouver_data(start_date, end_date)
    assert isinstance(result.loc["2010-01-05"], pd.Series), "Date supposed to exist, please verify function"