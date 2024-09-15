import pandas as pd

def load_data(file_path, encoding='ISO-8859-1', delimiter=';'):
    """
    Loads the dataset from a specified file path.

    Parameters:
    - file_path (str): Path to the CSV file.
    - encoding (str): Encoding of the CSV file.
    - delimiter (str): Delimiter used in the CSV file.

    Returns:
    - pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def inspect_data(data):
    """
    Prints basic information about the dataset.

    Parameters:
    - data (pd.DataFrame): The dataset to inspect.
    """
    print("Dataset Information:")
    print(data.info())

    print("\nFirst few rows of the dataset:")
    print(data.head())

    print("\nStatistical Summary:")
    print(data.describe())