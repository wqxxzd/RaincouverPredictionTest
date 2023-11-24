import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV

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
    
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Calculate and return the mean and standard deviation of cross-validation scores.

    This function performs cross-validation on the given model using the provided training data.
    It returns the mean and standard deviation of the scores for each metric.

    Parameters
    ----------
    model : estimator object
        The object to use to fit the data. 

    X_train : array-like of shape (n_samples, n_features)
        Training input samples. Could be a NumPy array, pandas DataFrame, or similar types.

    y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values.

    **kwargs : dict, optional
        Additional keyword arguments to pass to `cross_validate`. 

    Returns
    -------
    pd.Series
        Pandas Series containing the mean and standard deviation of cross-validation scores for each metric.
        The format of each entry is 'mean (+/- std)'.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    >>> model = RandomForestClassifier(random_state=42)
    >>> mean_std_cross_val_scores(model, X, y, cv=5)
    """
    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

def encode(data, col, max_val):
    """
    Encode a cyclical feature using sine and cosine transformations.

    This function adds two new columns to the input DataFrame, encoding
    a cyclical feature (like months or hours) using sine and cosine transformations.
    This method is based on the concept that cyclical data requires a representation
    where the end of the cycle connects back to the start seamlessly: 
    https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data to be transformed.
    col : str
        The name of the column in the DataFrame containing the cyclical feature.
    max_val : int or float
        The maximum value of the cyclical feature, used to scale the sine and cosine transformations.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with two new columns, encoding the original cyclical feature:
        - '{col}_sin': Sine transformation of the feature.
        - '{col}_cos': Cosine transformation of the feature.
        
    Examples:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'month': [1, 4, 7, 10]})
    >>> transformed_data = encode(data, 'month', 12)
    >>> print(transformed_data)
    
    Notes:
    -----
    This encoding technique is particularly useful for deep learning models that
    require understanding of the cyclical nature of certain features. It ensures
    continuity in the feature representation, acknowledging that after the highest
    value, the cycle starts again from the lowest.
    
    """
    if max_val <= 0:
        raise ValueError("max_val must be positive")

    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data