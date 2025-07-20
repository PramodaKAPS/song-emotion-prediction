import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import numpy as np
import pandas as pd

def assign_quadrant(arousal, valence):
    """Assigns quadrant based on arousal and valence."""
    if arousal >= 0 and valence >= 0:
        return 'Happy/Excited'
    elif arousal >= 0 and valence < 0:
        return 'Angry/Stressed'
    elif arousal < 0 and valence >= 0:
        return 'Calm/Peaceful'
    else:
        return 'Sad/Depressed'

def create_mlp_regressor(input_dim):
    """Creates a simple MLP regressor model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(X, y_a, y_v, epochs=10, drive_folder=None):
    """Trains models, computes metrics, and saves them."""
    arousal_model = create_mlp_regressor(X.shape[1])
    arousal_history = arousal_model.fit(X, y_a, epochs=epochs, verbose=1, validation_split=0.2)

    valence_model = create_mlp_regressor(X.shape[1])
    valence_history = valence_model.fit(X, y_v, epochs=epochs, verbose=1, validation_split=0.2)

    # Predict on training data
    y_a_pred = arousal_model.predict(X).flatten()
    y_v_pred = valence_model.predict(X).flatten()

    # Metrics
    a_mse = mean_squared_error(y_a, y_a_pred)
    a_r2 = r2_score(y_a, y_a_pred)
    v_mse = mean_squared_error(y_v, y_v_pred)
    v_r2 = r2_score(y_v, y_v_pred)

    # F1 via quadrants
    true_quad = [assign_quadrant(a, v) for a, v in zip(y_a, y_v)]
    pred_quad = [assign_quadrant(a, v) for a, v in zip(y_a_pred, y_v_pred)]
    f1 = f1_score(true_quad, pred_quad, average='macro')

    print(f"Training Metrics:\nArousal MSE: {a_mse:.4f}, R²: {a_r2:.4f}\nValence MSE: {v_mse:.4f}, R²: {v_r2:.4f}\nF1 Score: {f1:.4f}")

    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Arousal MSE', 'Arousal R²', 'Valence MSE', 'Valence R²', 'F1 Score'],
        'Value': [a_mse, a_r2, v_mse, v_r2, f1]
    })
    metrics_df.to_csv(os.path.join(drive_folder, 'training_metrics.csv'), index=False)
    print("Metrics saved.")

    return arousal_model, valence_model
