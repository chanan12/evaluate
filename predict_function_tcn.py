import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from tcn import TCN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_tcn_model(num_features, timesteps):
    inputs = Input(shape=(timesteps, num_features))
    x = TCN(
        nb_filters=64,
        kernel_size=3,
        dilations=[1, 2],  # Reduced the number of dilations
        padding='causal',
        activation='relu',
        return_sequences=False  # Return only the last timestep
    )(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_model(num_features, timesteps=30):
    m = create_tcn_model(num_features=num_features, timesteps=timesteps)
    return m

def create_sequences(X, y, timesteps):
    if len(X) < timesteps:
        return np.array([]), np.array([])
    sequences_X = []
    sequences_y = []
    for i in range(len(X) - timesteps + 1):
        seq_X = X[i:i + timesteps]
        seq_y = y[i + timesteps - 1]
        sequences_X.append(seq_X)
        sequences_y.append(seq_y)
    return np.array(sequences_X), np.array(sequences_y)

def scale_and_reshape(scaler, data, num_samples, timesteps, num_features):
    data_flat = data.reshape(num_samples * timesteps, num_features)
    data_scaled = scaler.transform(data_flat).reshape(num_samples, timesteps, num_features)
    return data_scaled

def train_and_predict_tcn(
        X_train,
        y_train,
        X_test,
        y_test,
        ticker,
        save_model_path,
        direction_threshold=0.005,
        epochs_per_day=1,
        batch_size=32,
        X_test_original=None,
        n_days=10
):
    """
    Trains the model day-by-day and returns:
    (model, test_predictions, daily_metrics, cv_mae, direction_accuracy, precision, recall, f1, cm, predicted_direction)
    """
    os.makedirs(save_model_path, exist_ok=True)
    model_file = os.path.join(save_model_path, f"tcn_model_{ticker}.h5")
    scaler_file = os.path.join(save_model_path, f"tcn_scaler_{ticker}.pkl")
    num_samples_train, timesteps, num_features = X_train.shape
    num_samples_test = X_test.shape[0]

    # Flatten for scaling
    X_train_flat = X_train.reshape(num_samples_train * timesteps, num_features)
    X_test_flat = X_test.reshape(num_samples_test * timesteps, num_features)

    # Split training into days
    samples_per_day = num_samples_train // n_days
    X_train_days = []
    y_train_days = []
    for d in range(n_days):
        start_idx = d * samples_per_day
        end_idx = start_idx + samples_per_day
        X_train_days.append(X_train[start_idx:end_idx])
        y_train_days.append(y_train[start_idx:end_idx])

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

    # Load or create model
    if os.path.exists(model_file):
        try:
            model = tf.keras.models.load_model(
                model_file,
                custom_objects={'TCN': TCN, 'mse': MeanSquaredError()}
            )
            # Recompile the model after loading
            model.compile(optimizer='adam', loss=MeanSquaredError())
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            logging.info("Creating a new model instead.")
            model = build_model(num_features=num_features, timesteps=timesteps)
    else:
        model = build_model(num_features=num_features, timesteps=timesteps)

    daily_metrics = []
    total_direction_matches = 0
    total_direction_days = 0

    # Daily incremental training and daily metrics
    for day_index in range(n_days):
        X_day = X_train_days[day_index]
        y_day = y_train_days[day_index]
        day_samples = X_day.shape[0]
        X_day_scaled = scale_and_reshape(scaler, X_day, day_samples, timesteps, num_features)

        # Train on today's data
        model.fit(X_day_scaled, y_day, epochs=epochs_per_day, batch_size=batch_size, verbose=0)

        # Compute daily metrics on test set
        X_test_scaled = scale_and_reshape(scaler, X_test, num_samples_test, timesteps, num_features)
        daily_predictions = model.predict(X_test_scaled).ravel()
        daily_mse = mean_squared_error(y_test, daily_predictions)
        daily_rmse = np.sqrt(daily_mse)
        daily_mae = mean_absolute_error(y_test, daily_predictions)

        # Compute daily direction match
        direction_match = None
        daily_precision = None
        daily_recall = None
        daily_f1 = None
        if day_index < n_days - 1:
            actual_today_price = y_train_days[day_index][-1]
            actual_tomorrow_price = y_train_days[day_index + 1][0]
            actual_action = 1 if actual_tomorrow_price > actual_today_price else 0
            last_sequence = X_day[-1].reshape(1, timesteps, num_features)
            last_sequence_scaled = scale_and_reshape(scaler, last_sequence, 1, timesteps, num_features)
            predicted_tomorrow_price = model.predict(last_sequence_scaled)[0, 0]
            predicted_action = 1 if predicted_tomorrow_price > actual_today_price else 0
            direction_match = 1 if predicted_action == actual_action else 0
            if direction_match == 1:
                total_direction_matches += 1
            total_direction_days += 1
            # With one sample, daily precision/recall/f1 are trivial
            TP = 1 if (actual_action == 1 and predicted_action == 1) else 0
            FP = 1 if (actual_action == 0 and predicted_action == 1) else 0
            FN = 1 if (actual_action == 1 and predicted_action == 0) else 0
            if (TP + FP) > 0:
                daily_precision = TP / (TP + FP)
            if (TP + FN) > 0:
                daily_recall = TP / (TP + FN)
            if daily_precision is not None and daily_recall is not None and (daily_precision + daily_recall) > 0:
                daily_f1 = 2 * (daily_precision * daily_recall) / (daily_precision + daily_recall)

        daily_metrics.append({
            'day': day_index + 1,
            'mse': daily_mse,
            'rmse': daily_rmse,
            'mae': daily_mae,
            'direction_match': direction_match,
            'precision': daily_precision,
            'recall': daily_recall,
            'f1': daily_f1
        })

    # Save the final model
    model.save(model_file, save_format="h5")

    # Cross-validation MAE
    cv_mae = None
    if len(X_train) >= 5:
        from scikeras.wrappers import KerasRegressor
        def cv_build_model():
            m = create_tcn_model(num_features=num_features, timesteps=timesteps)
            m.compile(optimizer='adam', loss=MeanSquaredError())  # Compile the model here
            return m
        X_train_scaled = scale_and_reshape(scaler, X_train, num_samples_train, timesteps, num_features)
        kr = KerasRegressor(build_fn=cv_build_model, epochs=5, batch_size=batch_size, verbose=0)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train_scaled):
            kr.fit(X_train_scaled[train_idx], y_train[train_idx])
            val_pred = kr.predict(X_train_scaled[val_idx])
            val_mae = mean_absolute_error(y_train[val_idx], val_pred)
            cv_scores.append(val_mae)
        cv_mae = np.mean(cv_scores)

    # Final predictions on test set and global direction metrics
    X_test_scaled = scale_and_reshape(scaler, X_test, num_samples_test, timesteps, num_features)
    test_predictions = model.predict(X_test_scaled).ravel()
    direction_accuracy = precision = recall = f1 = cm = predicted_direction = None
    if (X_test_original is not None) and ('actual_price' in X_test_original.columns) and (len(X_test_original) == len(y_test)):
        current_prices = X_test_original['actual_price'].values
        actual_change_ratio = (y_test - current_prices) / current_prices
        predicted_change_ratio = (test_predictions - current_prices) / current_prices
        actual_direction = np.where(
            actual_change_ratio > direction_threshold, 1,
            np.where(actual_change_ratio < -direction_threshold, 0, np.nan)
        )
        predicted_direction = np.where(
            predicted_change_ratio > direction_threshold, 1,
            np.where(predicted_change_ratio < -direction_threshold, 0, np.nan)
        )
        mask = ~np.isnan(actual_direction) & ~np.isnan(predicted_direction)
        filtered_actual_direction = actual_direction[mask].astype(int)
        filtered_predicted_direction = predicted_direction[mask].astype(int)
        if len(filtered_actual_direction) == 0:
            logging.warning("No significant movements found after applying direction threshold. No direction metrics computed.")
        else:
            direction_correct = np.sum(filtered_predicted_direction == filtered_actual_direction)
            direction_accuracy = direction_correct / len(filtered_actual_direction)
            unique_classes = np.unique(filtered_actual_direction)
            if unique_classes.size == 1:
                if np.array_equal(filtered_actual_direction, filtered_predicted_direction):
                    precision = recall = f1 = 1.0
                else:
                    precision = recall = f1 = 0.0
                cm = confusion_matrix(filtered_actual_direction, filtered_predicted_direction)
            else:
                precision = precision_score(filtered_actual_direction, filtered_predicted_direction, zero_division=0)
                recall = recall_score(filtered_actual_direction, filtered_predicted_direction, zero_division=0)
                f1 = f1_score(filtered_actual_direction, filtered_predicted_direction, zero_division=0)
                cm = confusion_matrix(filtered_actual_direction, filtered_predicted_direction)
    else:
        logging.warning("No 'actual_price' found or mismatch in data. Cannot compute global direction metrics.")

    # Return in the requested order:
    # (model, test_predictions, daily_metrics, cv_mae, direction_accuracy, precision, recall, f1, cm, predicted_direction)
    return (
        model,
        test_predictions,
        daily_metrics,
        cv_mae,
        direction_accuracy,
        precision,
        recall,
        f1,
        cm,
        predicted_direction
    )