import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = []

    def evaluate_model(self, model, model_name, params):
        """
        Evaluate the model using MAE and MAPE and store the results.
        """
        model.set_params(**params)
        model.fit(self.X_train, self.y_train)

        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(self.y_test, y_test_pred)

        self.results.append({
            'Model': model_name,
            'Parameters': params,
            'Train MAE': train_mae,
            'Train MAPE': train_mape,
            'Test MAE': test_mae,
            'Test MAPE': test_mape,
        })

    def train_knn(self):
        """
        Trains KNN regressor with different hyperparameters.
        """
        knn = KNeighborsRegressor()
        param_grid = [{'n_neighbors': k} for k in [3, 5, 7]]
        for params in param_grid:
            self.evaluate_model(knn, 'KNN', params)

    def train_decision_tree(self):
        """
        Trains Decision Tree regressor with different hyperparameters.
        """
        dt = DecisionTreeRegressor(random_state=42)
        param_grid = [{'max_depth': d, 'min_samples_leaf': l} for d in [5, 10] for l in [2, 4]]
        for params in param_grid:
            self.evaluate_model(dt, 'Decision Tree', params)

    def train_random_forest(self):
        """
        Trains Random Forest regressor with different hyperparameters.
        """
        rf = RandomForestRegressor(random_state=42)
        param_grid = [{'n_estimators': n, 'max_depth': d} for n in [50, 100] for d in [10, None]]
        for params in param_grid:
            self.evaluate_model(rf, 'Random Forest', params)

    def train_ann(self):
        """
        Trains ANN regressor with different hyperparameters.
        """
        ann = MLPRegressor(max_iter=500, random_state=42)
        param_grid = [{'hidden_layer_sizes': h, 'activation': a} for h in [(50,), (100,)] for a in ['relu', 'tanh']]
        for params in param_grid:
            self.evaluate_model(ann, 'ANN', params)

    def get_results(self):
        """
        Returns the model results as a pandas DataFrame.
        """
        results_df = pd.DataFrame(self.results)

        mean_train_mae = results_df['Train MAE'].mean()
        mean_train_mape = results_df['Train MAPE'].mean()
        mean_test_mae = results_df['Test MAE'].mean()
        mean_test_mape = results_df['Test MAPE'].mean()

        benchmark_row = pd.DataFrame([{
            'Model': 'Benchmark',
            'Parameters': '-',
            'Train MAE': mean_train_mae,
            'Train MAPE': mean_train_mape,
            'Test MAE': mean_test_mae,
            'Test MAPE': mean_test_mape
        }])

        results_df = pd.concat([results_df, benchmark_row], ignore_index=True)

        return results_df