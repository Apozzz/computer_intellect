import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce

class Preprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.high_cardinality_cols = [col for col in self.categorical_cols if self.data[col].nunique() > 100]
        self.categorical_cols = [col for col in self.categorical_cols if col not in self.high_cardinality_cols]
        self.freq_encodings = {}

    def handle_missing_values(self, strategy='median'):
        if strategy == 'mean':
            for col in self.numerical_cols:
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            for col in self.categorical_cols:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        elif strategy == 'median':
            for col in self.numerical_cols:
                self.data[col] = self.data[col].fillna(self.data[col].median())
            for col in self.categorical_cols:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        elif strategy == 'mode':
            for col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        elif strategy == 'drop':
            self.data.dropna(inplace=True)
        else:
            raise ValueError("Invalid strategy provided.")

        self.optimize_memory_usage()

    def handle_missing_values_post_encoding(self):
        """Handles missing values after encoding."""
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)

    def encode_categorical_variables(self):
        self.data = pd.get_dummies(self.data, columns=self.categorical_cols, drop_first=True)

    def encode_high_cardinality_variables(self):
        encoder = ce.TargetEncoder(cols=self.high_cardinality_cols)
        self.data[self.high_cardinality_cols] = encoder.fit_transform(
            self.data[self.high_cardinality_cols], self.data['price']
        )
        self.target_encoder = encoder

    def standardize_numerical_data(self, target_column):
        """Standardizes numerical features using StandardScaler, excluding the target variable."""
        features_to_scale = [col for col in self.numerical_cols if col != target_column]
        scaler = StandardScaler()
        self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])

    def get_preprocessed_data(self):
        """Returns the preprocessed dataset."""
        return self.data

    def split_data(self, target_column, test_size=0.2, random_state=42):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def optimize_memory_usage(self):
        """Optimizes memory usage by downcasting data types."""
        for col in self.numerical_cols:
            col_min = self.data[col].min()
            col_max = self.data[col].max()
            if pd.api.types.is_float_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], downcast='float')
            elif pd.api.types.is_integer_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], downcast='integer')

    def remove_irrelevant_features(self):
        irrelevant_cols = ['id', 'title', 'body', 'source', 'has_photo', 'time']
        self.data.drop(columns=irrelevant_cols, inplace=True, errors='ignore')
        self.numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.high_cardinality_cols = [col for col in self.categorical_cols if self.data[col].nunique() > 100]
        self.categorical_cols = [col for col in self.categorical_cols if col not in self.high_cardinality_cols]

    def remove_outliers(self):
        numerical_cols = ['price', 'square_feet', 'bathrooms', 'bedrooms']
        filter = pd.Series(True, index=self.data.index)
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            col_filter = (self.data[col] >= Q1 - 1.5 * IQR) & (self.data[col] <= Q3 + 1.5 * IQR)
            filter &= col_filter
        self.data = self.data.loc[filter]
    
    def scale_numerical_features(self, target_column):
        features_to_scale = [col for col in self.numerical_cols if col != target_column]
        scaler = MinMaxScaler()
        self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])
        self.scaler = scaler

    def preprocess_data(self, target_column):
        self.remove_irrelevant_features()
        self.handle_missing_values(strategy='median')
        self.encode_categorical_variables()
        self.encode_high_cardinality_variables()
        self.remove_outliers()
        self.scale_numerical_features(target_column)
        self.data[target_column] = pd.to_numeric(self.data[target_column], errors='coerce')
        self.data.dropna(subset=[target_column], inplace=True)
