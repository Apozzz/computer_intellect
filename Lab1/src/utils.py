from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns MAE and MAPE.

    Parameters:
    - model: Trained model.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test target.

    Returns:
    - mae (float): Mean Absolute Error.
    - mape (float): Mean Absolute Percentage Error.
    - y_pred (np.array): Predicted values.
    """
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mae, mape, y_pred

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """
    Plots actual vs. predicted values.

    Parameters:
    - y_test (pd.Series): Actual target values.
    - y_pred (pd.Series): Predicted target values.
    - model_name (str): Name of the model.
    """
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_test.index)

    if len(y_test) > 1000:
        y_test_sample = y_test.sample(n=1000, random_state=42)
        y_pred_sample = y_pred.loc[y_test_sample.index]
    else:
        y_test_sample = y_test
        y_pred_sample = y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_sample, y_pred_sample, alpha=0.5, color='blue')
    plt.plot(
        [y_test_sample.min(), y_test_sample.max()],
        [y_test_sample.min(), y_test_sample.max()],
        'r--'
    )
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Actual vs. Predicted Prices ({model_name})')
    plt.show()