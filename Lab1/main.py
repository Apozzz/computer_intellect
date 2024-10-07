from src.data_loader import load_data, inspect_data
from src.data_quality import DataQualityReport
from src.preprocessing import Preprocessor
from src.utils import evaluate_model, plot_actual_vs_predicted, sample_and_visualize, combined_boxplots
from dotenv import load_dotenv
import os
import pandas as pd
from src.models import ModelTrainer

def main():
    pd.options.display.float_format = '{:.2f}'.format
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    load_dotenv()
    csv_file_path = os.getenv("EXTRACTED_DATA_PATH")

    data = load_data(csv_file_path)
    if data is not None:
        print("Data loaded successfully.")
        inspect_data(data)

        report = DataQualityReport(data)
        report.get_data_types()
        report.check_missing_values()
        report.outliers_summary()
        report.continuous_summary()
        report.categorical_summary(max_categories=20)

        sample_and_visualize(data)
        combined_boxplots(data)

        target_column = 'price'

        preprocessor = Preprocessor(data)
        preprocessor.preprocess_for_reporting(target_column)
        data_preprocessed_report = preprocessor.get_preprocessed_data()

        report = DataQualityReport(data_preprocessed_report)
        report.get_data_types()
        report.check_missing_values()
        report.outliers_summary()
        report.continuous_summary()
        report.categorical_summary(max_categories=20)

        sample_and_visualize(data_preprocessed_report)
        combined_boxplots(data_preprocessed_report)

        preprocessor.preprocess_data(target_column, scaling_method=None)
        data_preprocessed = preprocessor.get_preprocessed_data()

        print("Checking for NaNs in preprocessed data...")
        nan_counts = data_preprocessed.isna().sum()
        print(nan_counts[nan_counts > 0])

        target_column = 'price'
        if target_column not in data_preprocessed.columns:
            print(f"Error: Target column '{target_column}' not found in preprocessed data.")
            return

        X_train, X_test, y_train, y_test = preprocessor.split_data(target_column)
        trainer = ModelTrainer(X_train, y_train, X_test, y_test)

        print("Checking for NaNs in X_train...")
        nan_counts_X_train = X_train.isna().sum()
        print(nan_counts_X_train[nan_counts_X_train > 0])

        print("Checking for NaNs in X_test...")
        nan_counts_X_test = X_test.isna().sum()
        print(nan_counts_X_test[nan_counts_X_test > 0])

        print("Checking for NaNs in y_train...")
        print(y_train.isna().sum())

        print("Checking for NaNs in y_test...")
        print(y_test.isna().sum())

        print("\nTraining KNN...")
        trainer.train_knn()

        print("\nTraining Decision Tree...")
        #trainer.train_decision_tree()

        print("\nTraining Random Forest...")
        #trainer.train_random_forest()

        print("\nTraining ANN...")
        #trainer.train_ann()
        
        results = trainer.get_results()
        print("\nModel Performance Comparison:")
        print(results)

    else:
        print("Failed to load data.")


if __name__ == "__main__":
    main()