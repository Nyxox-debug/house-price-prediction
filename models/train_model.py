from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_and_save_model(X, y, model_path='house_price_model.pkl', test_size=0.2, random_state=42):
    """
    Train a Linear Regression model and save it.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        model_path (str): Path to save the trained model.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for reproducibility.
    Returns:
        model: Trained model.
        X_test, y_test, y_pred: Test data and predictions for evaluation.
        mse (float): Mean Squared Error on test set.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Save model
    joblib.dump(model, model_path)
    
    return model, X_test, y_test, y_pred, mse
