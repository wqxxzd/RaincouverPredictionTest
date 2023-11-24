import numpy as np

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
    # Encode the feature using sine and cosine transformations

    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data