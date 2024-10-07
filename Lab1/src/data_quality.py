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

    def outliers_summary(self):
        """Identifies outliers in numerical features using the IQR method."""
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.data[(self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))]
            print(f"\nOutliers in {col}: {len(outliers)}")

    def continuous_summary(self):
        """
        Generates a summary table for continuous (numerical) features.
        """
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        summary = pd.DataFrame(index=numerical_cols)
        summary['Count'] = self.data[numerical_cols].count()
        summary['% Miss'] = self.data[numerical_cols].isnull().mean() * 100
        summary['Card.'] = self.data[numerical_cols].nunique()
        summary['Min'] = self.data[numerical_cols].min()
        summary['Q1'] = self.data[numerical_cols].quantile(0.25)
        summary['Mean'] = self.data[numerical_cols].mean()
        summary['Median'] = self.data[numerical_cols].median()
        summary['Q3'] = self.data[numerical_cols].quantile(0.75)
        summary['Max'] = self.data[numerical_cols].max()
        summary['Std. Dev.'] = self.data[numerical_cols].std()

        print("\nContinuous Feature Summary:")
        print(summary)

    def categorical_summary(self, max_categories=20):
        """
        Generates a summary table for categorical features.
        Only shows the top categories and avoids overloading visualizations for high cardinality.
        """
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        summary = pd.DataFrame(index=categorical_cols)
        summary['Count'] = self.data[categorical_cols].count()
        summary['% Miss'] = self.data[categorical_cols].isnull().mean() * 100
        summary['Card.'] = self.data[categorical_cols].nunique()

        modes = []
        mode_freqs = []
        mode_2nd = []
        mode_2nd_freqs = []
        mode_freq_percent = []
        mode_2nd_freq_percent = []

        for col in categorical_cols:
            value_counts = self.data[col].value_counts()
            modes.append(value_counts.index[0])
            mode_freqs.append(value_counts.iloc[0])
            mode_freq_percent.append((value_counts.iloc[0] / summary['Count'][col]) * 100)
            
            if len(value_counts) > 1:
                mode_2nd.append(value_counts.index[1])
                mode_2nd_freqs.append(value_counts.iloc[1])
                mode_2nd_freq_percent.append((value_counts.iloc[1] / summary['Count'][col]) * 100)
            else:
                mode_2nd.append(None)
                mode_2nd_freqs.append(None)
                mode_2nd_freq_percent.append(None)

        summary['Mode'] = modes
        summary['Mode Freq'] = mode_freqs
        summary['Mode %'] = mode_freq_percent
        summary['2nd Mode'] = mode_2nd
        summary['2nd Mode Freq'] = mode_2nd_freqs
        summary['2nd Mode %'] = mode_2nd_freq_percent

        print("\nCategorical Feature Summary:")
        print(summary)