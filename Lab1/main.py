from src.data_loader import load_data, inspect_data
from src.data_quality import DataQualityReport
from src.preprocessing import Preprocessor
from src.models import train_knn, train_decision_tree, train_random_forest, train_ann
from src.utils import evaluate_model, plot_actual_vs_predicted
from dotenv import load_dotenv
import os
import pandas as pd

def main():
    load_dotenv()
    csv_file_path = os.getenv("EXTRACTED_DATA_PATH")

    data = load_data(csv_file_path)
    if data is not None:
        print("Data loaded successfully.")
        inspect_data(data)

        report = DataQualityReport(data)
        report.get_data_types()
        report.check_missing_values()
        report.categorical_summary()
        report.numerical_summary()
        report.outliers_summary()

        target_column = 'price'

        preprocessor = Preprocessor(data)
        preprocessor.preprocess_data(target_column)
        data_preprocessed = preprocessor.get_preprocessed_data()

        # Check for NaNs
        print("Checking for NaNs in preprocessed data...")
        nan_counts = data_preprocessed.isna().sum()
        print(nan_counts[nan_counts > 0])

        target_column = 'price'
        if target_column not in data_preprocessed.columns:
            print(f"Error: Target column '{target_column}' not found in preprocessed data.")
            return

        X_train, X_test, y_train, y_test = preprocessor.split_data(target_column)

        # Check for NaNs in X_train and X_test
        print("Checking for NaNs in X_train...")
        nan_counts_X_train = X_train.isna().sum()
        print(nan_counts_X_train[nan_counts_X_train > 0])

        print("Checking for NaNs in X_test...")
        nan_counts_X_test = X_test.isna().sum()
        print(nan_counts_X_test[nan_counts_X_test > 0])

        # Check for NaNs in y_train and y_test
        print("Checking for NaNs in y_train...")
        print(y_train.isna().sum())

        print("Checking for NaNs in y_test...")
        print(y_test.isna().sum())

        # TRAIN
        print("\nTraining KNN model...")
        best_knn, grid_search_knn = train_knn(X_train, y_train)
        print(f"Best KNN Parameters: {grid_search_knn.best_params_}")

        print("\nTraining Decision Tree model...")
        best_dt, grid_search_dt = train_decision_tree(X_train, y_train)
        print(f"Best Decision Tree Parameters: {grid_search_dt.best_params_}")

        print("\nTraining Random Forest model...")
        best_rf, grid_search_rf = train_random_forest(X_train, y_train)
        print(f"Best Random Forest Parameters: {grid_search_rf.best_params_}")

        print("\nTraining ANN model...")
        best_ann, grid_search_ann = train_ann(X_train, y_train)
        print(f"Best ANN Parameters: {grid_search_ann.best_params_}")

        # EVALUATE
        print("\nEvaluating KNN model...")
        mae_knn, mape_knn, y_pred_knn = evaluate_model(best_knn, X_test, y_test)
        print(f"KNN MAE: {mae_knn:.2f}, MAPE: {mape_knn:.2%}")
        plot_actual_vs_predicted(y_test, y_pred_knn, 'KNN')

        print("\nEvaluating Decision Tree model...")
        mae_dt, mape_dt, y_pred_dt = evaluate_model(best_dt, X_test, y_test)
        print(f"Decision Tree MAE: {mae_dt:.2f}, MAPE: {mape_dt:.2%}")
        plot_actual_vs_predicted(y_test, y_pred_dt, 'Decision Tree')

        print("\nEvaluating Random Forest model...")
        mae_rf, mape_rf, y_pred_rf = evaluate_model(best_rf, X_test, y_test)
        print(f"Random Forest MAE: {mae_rf:.2f}, MAPE: {mape_rf:.2%}")
        plot_actual_vs_predicted(y_test, y_pred_rf, 'Random Forest')

        print("\nEvaluating ANN model...")
        mae_ann, mape_ann, y_pred_ann = evaluate_model(best_ann, X_test, y_test)
        print(f"ANN MAE: {mae_ann:.2f}, MAPE: {mape_ann:.2%}")
        plot_actual_vs_predicted(y_test, y_pred_ann, 'ANN')

        # COMPILE
        results = pd.DataFrame({
            'Model': ['KNN', 'Decision Tree', 'Random Forest', 'ANN'],
            'MAE': [mae_knn, mae_dt, mae_rf, mae_ann],
            'MAPE': [mape_knn, mape_dt, mape_rf, mape_ann]
        })

        print("\nModel Performance Comparison:")
        print(results)

    else:
        print("Failed to load data.")


if __name__ == "__main__":
    main()