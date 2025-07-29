# House Price Prediction

A beginner-friendly ML project to predict house prices using Linear Regression.

## Description

This project uses the California Housing dataset to train a Linear Regression model, predicting house values based on features like median income and house age. It includes functionality to predict prices for new houses using a saved model. The code is modular, with separate files for data loading, model training, visualization, and prediction.

## Requirements

- Python 3.8+
- Install dependencies: `uv pip install -r requirements.txt`

## Setup

1. Clone the repository: `git clone <your-repo-url>`
2. Create a virtual environment: `uv venv`
3. Activate: `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows)
4. Install dependencies: `uv pip install -r requirements.txt`
5. Run: `python main.py`

## Usage

- Train and evaluate the model: `python main.py`
- Predict for a new house: Modify `main.py` with new `MedInc` and `HouseAge` values or call `predict_new_house(med_inc, house_age)` in a script.

## Outputs

- `house_price_model.pkl`: Trained model.
- `price_prediction.png`: Plot of actual vs. predicted house values.

## Dataset

California Housing dataset (built into scikit-learn).

## Example Prediction

```python
from predict.predict import predict_new_house
price = predict_new_house(med_inc=8.0, house_age=20.0)
print(f'Predicted price: ${price*100000:.2f}')
```
