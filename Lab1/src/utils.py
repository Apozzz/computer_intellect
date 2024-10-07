from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

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

    if len(y_test) > 100000000:
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

def sample_and_visualize(data, max_samples=100000):
    """
    Takes a sample from the dataset and visualizes numerical distributions.
    Adjusts the display of large values (e.g., ID, Time).
    """
    if len(data) > max_samples:
        sample_data = data.sample(n=max_samples, random_state=42)
        print(f"Using a sample of {max_samples} rows for visualization.")
    else:
        sample_data = data

    numerical_data = sample_data.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

    if numerical_data.empty:
        print("No numerical columns to visualize.")
        return

    columns_to_plot = ['id', 'price', 'square_feet', 'bathrooms', 'bedrooms', 'latitude', 'longitude', 'time']

    columns_to_plot = [col for col in columns_to_plot if col in numerical_data.columns]

    if not columns_to_plot:
        print("No columns available for plotting after filtering.")
        return

    num_features = len(columns_to_plot)
    num_cols = 3
    num_rows = (num_features // num_cols) + (num_features % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        numerical_data[col].hist(ax=axes[i], bins=20)
        axes[i].set_title(f'Histogram of {col}')
    
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def combined_boxplots(data, max_cols=3, exclude_outliers=False):
    """
    Plots box plots for all numerical features in one figure, arranged in a grid layout.
    Can exclude outliers for clearer visual interpretation.
    max_cols: The maximum number of columns in the grid.
    exclude_outliers: If True, outliers are excluded from boxplots.
    """
    numerical_cols = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    num_features = len(numerical_cols)
    max_rows = math.ceil(num_features / max_cols)

    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(15, max_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        if exclude_outliers:
            sns.boxplot(data=data, y=col, ax=axes[i], showfliers=False)
        else:
            sns.boxplot(data=data, y=col, ax=axes[i])
        axes[i].set_title(f'Box Plot: {col}')
    
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()