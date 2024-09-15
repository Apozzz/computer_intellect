from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_knn(X_train, y_train):
    """
    Trains a KNN regressor using GridSearchCV to find the best k.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - best_knn: Best estimator from GridSearchCV.
    - grid_search_knn: The GridSearchCV object.
    """
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    knn = KNeighborsRegressor()
    grid_search_knn = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search_knn.fit(X_train, y_train)
    best_knn = grid_search_knn.best_estimator_
    return best_knn, grid_search_knn

def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree regressor using GridSearchCV.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - best_dt: Best estimator from GridSearchCV.
    - grid_search_dt: The GridSearchCV object.
    """
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6]
    }
    dt = DecisionTreeRegressor(random_state=42)
    grid_search_dt = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search_dt.fit(X_train, y_train)
    best_dt = grid_search_dt.best_estimator_
    return best_dt, grid_search_dt

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest regressor using GridSearchCV.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - best_rf: Best estimator from GridSearchCV.
    - grid_search_rf: The GridSearchCV object.
    """
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    randomized_search_rf = RandomizedSearchCV(
        rf, 
        param_grid, 
        n_iter=10, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1, 
        random_state=42,
        verbose=2,
    )
    randomized_search_rf.fit(X_train, y_train)
    best_rf = randomized_search_rf.best_estimator_
    return best_rf, randomized_search_rf

def train_ann(X_train, y_train):
    """
    Trains an ANN regressor using GridSearchCV.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - best_ann: Best estimator from GridSearchCV.
    - grid_search_ann: The GridSearchCV object.
    """
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'learning_rate': ['constant', 'adaptive']
    }
    ann = MLPRegressor(max_iter=500, random_state=42)
    randomized_search_ann = RandomizedSearchCV(
        estimator=ann,
        param_distributions=param_dist,
        n_iter=6,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )
    randomized_search_ann.fit(X_train, y_train)
    best_ann = randomized_search_ann.best_estimator_
    return best_ann, randomized_search_ann