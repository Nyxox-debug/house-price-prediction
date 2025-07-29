import pandas as pd
import joblib
import numpy as np

def predict_new_house(model_path='house_price_model.pkl', med_inc=None, house_age=None):
    """
    Predict the price of a new house using the trained model.
    Args:
        model_path (str): Path to the saved model.
        med_inc (float): Median income in block group (in tens of thousands of dollars).
        house_age (float): Median house age in years.
    Returns:
        float: Predicted house value (in $100,000s).
    """
    # Load the trained model
    model = joblib.load(model_path)
    
    # Prepare input data as a DataFrame (matching training features)
    input_data = pd.DataFrame([[med_inc, house_age]], columns=['MedInc', 'HouseAge'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return prediction
