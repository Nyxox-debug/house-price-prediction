import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_housing_data():
    """
    Load the California Housing dataset and return features and target.
    Returns:
        X (pd.DataFrame): Features (e.g., median income, house age).
        y (pd.Series): Target (median house value in $100,000s).
    """
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='MedHouseVal')
    return X, y
