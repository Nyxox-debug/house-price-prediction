from data.load_data import load_housing_data
from models.train_model import train_and_save_model
from visualization.plot_results import plot_predictions
from predict.predict import predict_new_house

def main():
    # Load data
    X, y = load_housing_data()
    
    # Select features (for simplicity and low resource usage)
    X = X[['MedInc', 'HouseAge']]  # Median income, house age
    
    # Train model
    model, X_test, y_test, y_pred, mse = train_and_save_model(X, y)
    
    # Print evaluation
    print(f'Mean Squared Error: {mse:.2f}')
    
    # Plot results
    plot_predictions(y_test, y_pred)
    print(f'Plot saved as price_prediction.png')
    
    # Predict for a new house (example values)
    new_med_inc = 8.0  # Example: $80,000 median income
    new_house_age = 20.0  # Example: 20 years old
    predicted_price = predict_new_house(med_inc=new_med_inc, house_age=new_house_age)
    print(f'Predicted price for a house with MedInc={new_med_inc}, HouseAge={new_house_age}: ${predicted_price*100000:.2f}')

if __name__ == '__main__':
    main()
