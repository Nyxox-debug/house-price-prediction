from data.load_data import load_housing_data
from models.train_model import train_and_save_model
from visualization.plot_results import plot_predictions

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

if __name__ == '__main__':
    main()
