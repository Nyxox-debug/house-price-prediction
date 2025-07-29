import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred, save_path='price_prediction.png'):
    """
    Create and save a scatter plot of actual vs. predicted prices.
    Args:
        y_test (pd.Series): Actual prices.
        y_pred (np.ndarray): Predicted prices.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual House Values ($100,000s)')
    plt.ylabel('Predicted House Values ($100,000s)')
    plt.title('Actual vs. Predicted House Values')
    plt.savefig(save_path)
    plt.close()
