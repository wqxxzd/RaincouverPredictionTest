#Below script is adpoted from:
#https://github.com/ttimbers/breast_cancer_predictor_py/blob/v2.0.0/scripts/split_n_preprocess.py

import click
import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import encode


@click.command()
@click.option('--data-file', type=str, help="Path to raw data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--seed', type=int, help="Random seed", default=522)

def main(data_file, data_to, preprocessor_to, seed):
    '''This script is to drop the features that are highly corrleted with 
    the selected features, to split the data into train and test sets, 
    as well as create the preprocessor for model training. '''
    
    # Create relevant file paths for outputs
    # write_to path transforming
    if data_to != '':
        data_to = data_to if data_to[-1] == '/' else data_to + '/'
            
    # Check write_to path existence
    if not os.path.exists(data_to):
        os.mkdir(data_to)

    # write_to path transforming
    if preprocessor_to != '':
        preprocessor_to = preprocessor_to if preprocessor_to[-1] == '/' else preprocessor_to + '/'
            
    # Check write_to path existence
    if not os.path.exists(preprocessor_to):
        os.mkdir(preprocessor_to)

    np.random.seed(seed)
    
    # Read the raw data file and remove features that are highly correlated with the selected 
    #features based on the concludsion from EDA
    precipit_df = pd.read_csv(data_file)
    precipit_df['is_precipitation'] = precipit_df['precipitation_sum'] > 0.01
    precipit_df['date'] = pd.to_datetime(precipit_df['date'])
    precipit_df['month'] = precipit_df['date'].dt.month
    
    precipit_df = precipit_df.drop(columns=['sunrise',
                                            'sunset', 
                                            'weather_code', 
                                            'rain_sum',
                                            'snowfall_sum',
                                            'date',
                                            'precipitation_hours', 
                                            'temperature_2m_max',
                                            'temperature_2m_min',
                                            'apparent_temperature_max', 
                                            'apparent_temperature_min', 
                                            'apparent_temperature_mean',
                                            'wind_gusts_10m_max',
                                            'precipitation_sum'])
    

    # Split data set into 80% training set and 20% test set
    train_df, test_df = train_test_split(
        precipit_df, test_size=0.2, random_state=seed
    )

    X_train = train_df.drop(columns=['is_precipitation'])
    y_train = train_df['is_precipitation']

    X_test = test_df.drop(columns=['is_precipitation'])
    y_test = test_df['is_precipitation']
    
    # Column transformation and preprocessing
    X_train = encode(X_train, 'month', 12)
    X_test = encode(X_test, 'month', 12 )

    
    numeric_transformer = StandardScaler()

    preprocess = make_column_transformer(
        (numeric_transformer, make_column_selector(dtype_include='number')),
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # Exporting X_train, y_train, X_test, y_test and preprocessor
    X_train.to_csv(os.path.join(data_to, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(data_to, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_to, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(data_to, "y_test.csv"), index=False)
    
    pickle.dump(preprocess, open(os.path.join(preprocessor_to, "precipit_preprocessor.pickle"), "wb"))

if __name__ == '__main__':
    main()