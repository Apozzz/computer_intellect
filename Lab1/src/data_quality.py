import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataQualityReport:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with the dataset.
        """
        self.data = data

    def get_data_types(self):
        """Prints data types of each column."""
        print("Data Types:")
        print(self.data.dtypes)

    def check_missing_values(self):
        """Checks and prints missing values in each column."""
        missing_values = self.data.isnull().sum()
        print("Missing Values in Each Column:")
        print(missing_values)

    def categorical_summary(self):
        """Provides summary statistics for categorical features."""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\nCategorical Feature: {col}")
            print(self.data[col].value_counts())

    def numerical_summary(self):
        """Provides summary statistics for numerical features."""
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        print("Statistical Summary for Numerical Features:")
        print(self.data[numerical_cols].describe())

    def outliers_summary(self):
        """Identifies outliers in numerical features using the IQR method."""
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.data[(self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))]
            print(f"\nOutliers in {col}: {len(outliers)}")