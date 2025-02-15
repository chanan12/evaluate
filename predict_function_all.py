from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import os
import pickle
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, confusion_matrix
)
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    VotingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


def train_and_evaluate_model_with_actual_price(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    ticker,
    direction_threshold=0.005 # for 0.5%
):
    """
    Train or load a model, perform hyperparameter search to improve performance,
    scale data for models that can benefit, and evaluate performance including direction-based metrics.
    Incorporates direction accuracy, precision, recall, F1, confusion matrix, and a minimal threshold for meaningful direction change.

    Assumptions:
    1. 'y_train' and 'y_test' are actual prices aligned with X_train and X_test.
    2. 'X_test' contains a column 'yesterday_actual_price' for direction calculation.
    3. Returns:
       (model, predictions_price, mse, rmse, mae, cv_mae, direction_accuracy, precision, recall, f1, cm)
    """

    # Set the model save path
    save_model_path = r'C:\Users\User\pythonProject\pythonProject\pythonProject\aizevin\pythonProject\04_Stock_PKL_Files'
    os.makedirs(save_model_path, exist_ok=True)

    model_file = os.path.join(save_model_path, f"{model_name}_{ticker}.pkl")

    # Check feature consistency
    train_features = list(X_train.columns)
    test_features = list(X_test.columns)
    if train_features != test_features:
        print(f"Feature mismatch for ticker {ticker}:")
        print(f"Training features: {train_features}")
        print(f"Test features: {test_features}")
        return (
            None, None, None,
            None, None, None,
            None, None, None, None, None
        )

    # Scale data
    scaler = StandardScaler()
    X_train_values = X_train.values
    X_test_values = X_test.values

    X_train_scaled = scaler.fit_transform(X_train_values)
    X_test_scaled = scaler.transform(X_test_values)

    # Load or train model
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded saved model {model_name} for ticker {ticker}")

        # If model features changed, refit
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != X_train.shape[1]:
            print(f"Retraining {model_name} for ticker {ticker} due to feature mismatch.")
            model.fit(X_train_scaled, y_train)
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
    else:
        print(f"Training a new {model_name} model for ticker {ticker} with hyperparameter search...")

        # Define hyperparameter distributions
        if model_name == 'xgboost':
            base_model = XGBRegressor(objective='reg:squarederror', random_state=42)
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }

        elif model_name == 'svr':
            base_model = SVR()
            param_distributions = {
                'C': [0.1, 1.0, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear', 'poly']
            }

        elif model_name == 'randomized_search_rf':
            base_model = RandomForestRegressor(random_state=42)
            param_distributions = {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }

        elif model_name == 'gradient_boosting':
            base_model = GradientBoostingRegressor(random_state=42)
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3, 5],
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }

        elif model_name == 'extra_trees':
            base_model = ExtraTreesRegressor(random_state=42)
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }

        elif model_name == 'lasso_regression':
            base_model = Lasso(max_iter=10000, random_state=42)
            param_distributions = {
                'alpha': np.logspace(-4, -1, 20)
            }

        elif model_name == 'bayesian_ridge':
            print("Bayesian Ridge not implemented here directly.")
            return (
                None, None, None,
                None, None, None,
                None, None, None, None, None
            )

        elif model_name == 'linear_regression':
            # LinearRegression has no major hyperparams
            base_model = LinearRegression()
            param_distributions = None

        elif model_name == 'decision_tree':
            base_model = DecisionTreeRegressor(random_state=42)
            param_distributions = {
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }

        elif model_name == 'ensemble_model':
            base_model = VotingRegressor(
                estimators=[
                    ('xgboost', XGBRegressor(
                        objective='reg:squarederror', random_state=42
                    )),
                    ('ridge', Ridge(alpha=10, solver='cholesky')),
                    ('gradient_boosting', GradientBoostingRegressor(random_state=42))
                ]
            )
            param_distributions = None

        else:
            print(f"Model {model_name} is not defined.")
            return (
                None, None, None,
                None, None, None,
                None, None, None, None, None
            )

        # Perform hyperparameter search if defined
        if param_distributions is not None:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=20,
                cv=5,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            print(f"Best parameters for {model_name}: {search.best_params_}")
        else:
            model = base_model
            model.fit(X_train_scaled, y_train)

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model {model_name} for ticker {ticker} saved in {model_file}.")

    # Predict
    try:
        predictions_price = model.predict(X_test_scaled)
    except Exception as e:
        print(f"Error predicting with {model_name} for {ticker}: {e}")
        return (
            None, None, None,
            None, None, None,
            None, None, None, None, None
        )

    # Regression metrics
    try:
        mse = mean_squared_error(y_test, predictions_price)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions_price)
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        return (
            None, None, None,
            None, None, None,
            None, None, None, None, None
        )

    # CV mae
    cv_mae = None
    if len(X_train_scaled) >= 5 and model_name not in ['svr','lasso_regression']:
        try:
            cv_scores = cross_val_score(
                model,
                X_train_scaled,
                y_train,
                cv=5,
                scoring='neg_mean_absolute_error'
            )
            cv_mae = -np.mean(cv_scores)
        except Exception as e:
            print(f"Error during cross-validation: {e}")

    # Direction metrics
    if 'yesterday_actual_price' in test_features:
        yesterday_actual_prices = X_test['yesterday_actual_price'].values
    else:
        print("No 'yesterday_actual_price' found in X_test. Cannot compute direction metrics.")
        return (
            model, predictions_price, mse, rmse, mae, cv_mae,
            None, None, None, None, None
        )

    # Compute actual vs. predicted direction based on threshold
    actual_direction = (((y_test - yesterday_actual_prices) / yesterday_actual_prices) > direction_threshold).astype(int)
    predicted_direction = (((predictions_price - yesterday_actual_prices) / yesterday_actual_prices) > direction_threshold).astype(int)

    direction_correct = np.sum(predicted_direction == actual_direction)
    total_predictions = len(predicted_direction)
    direction_accuracy = direction_correct / total_predictions

    # Compute precision, recall, f1
    if np.unique(actual_direction).size == 1:
        # All actual directions are the same -> no precision/recall/f1
        precision = recall = f1 = None
        print("All actual directions are the same. Precision/Recall/F1 not defined.")
        cm = None
    else:
        precision = precision_score(actual_direction, predicted_direction, zero_division=0)
        recall = recall_score(actual_direction, predicted_direction, zero_division=0)
        f1 = f1_score(actual_direction, predicted_direction, zero_division=0)
        cm = confusion_matrix(actual_direction, predicted_direction)

    print(f"Direction Accuracy: {direction_accuracy}")
    if precision is not None:
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
        print("Confusion Matrix:")
        print(cm)

    return \
    (
        model,
        predictions_price,
        mse,
        rmse,
        mae,
        cv_mae,
        direction_accuracy,
        precision,
        recall,
        f1,
        cm,
        predicted_direction
    )
