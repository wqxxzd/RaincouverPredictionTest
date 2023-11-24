import numpy as np
def encode(data, col, max_val):
    # Using ordinal encoding  or leaving the month as a number may not be the best representation of month. 
    # Since month is a cyclical data, we get inspiration from this work and use cyclical encoding: 
    # https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
    # This preprocessing does not break golden rule because the max value of months is 12 as per common knowledge. 
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data