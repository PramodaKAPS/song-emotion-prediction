# model.py: Training, prediction, and metrics

import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import numpy as np

def create_mlp_regressor(input_dim):
    """Creates a multi-layer perceptron regressor model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_nn_models(X, y_a, y_v, epochs=10):
    """Trains MLP regressors for arousal and valence."""
    arousal_model = create_mlp_regressor(X.shape[1])
    arousal_model.fit(X, y_a, epochs=epochs, verbose=1, validation_split=0.2)

    valence_model = create_mlp_regressor(X.shape[1])
    valence_model.fit(X, y_v, epochs=epochs, verbose=1, validation_split=0.2)
    return arousal_model, valence_model

def compute_metrics(y_true_a, y_pred_a, y_true_v, y_pred_v):
    """Computes training metrics: MSE, R², and F1 (quadrant-based)."""
    arousal_mse = mean_squared_error(y_true_a, y_pred_a)
    arousal_r2 = r2_score(y_true_a, y_pred_a)
    valence_mse = mean_squared_error(y_true_v, y_pred_v)
    valence_r2 = r2_score(y_true_v, y_pred_v)
    # Quadrant-based F1
    true_quadrants = [assign_quadrant(a, v) for a, v in zip(y_true_a, y_true_v)]
    pred_quadrants = [assign_quadrant(a, v) for a, v in zip(y_pred_a, y_pred_v)]
    f1 = f1_score(true_quadrants, pred_quadrants, average='macro')
    return arousal_mse, arousal_r2, valence_mse, valence_r2, f1

