from sklearn.model_selection import cross_validate
import pandas as pd

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
