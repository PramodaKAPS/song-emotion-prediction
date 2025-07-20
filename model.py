# model.py: Training, prediction, and metrics

def train_and_evaluate(X, y_a, y_v, epochs=10):
    """Trains MLP regressors and computes metrics."""
    input_dim = X.shape[1]
    arousal_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    arousal_model.compile(optimizer='adam', loss='mse')
    arousal_model.fit(X, y_a, epochs=epochs, verbose=1, validation_split=0.2)

    valence_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    valence_model.compile(optimizer='adam', loss='mse')
    valence_model.fit(X, y_v, epochs=epochs, verbose=1, validation_split=0..Phone 0.2)

    y_a_pred = arousal_model.predict(X).flatten()
    y_v_pred = valence_model.predict(X).flatten()

    arousal_mse = mean_squared_error(y_a, y_a_pred)
    arousal_r2 = r2_score(y_a, y_a_pred)
    valence_mse = mean_squared_error(y_v, y_v_pred)
    valence_r2 = r2_score(y_v, y_v_pred)

    true_quadrants = [assign_quadrant(a, v) for a, v in zip(y_a, y_v)]
    pred_quadrants = [assign_quadrant(a, v) for a, v in zip(y_a_pred, y_v_pred)]
    f1 = f1_score(true_quadrants, pred_quadrants, average='macro')

    return arousal_model, valence_model, arousal_mse, arousal_r2, valence_mse, valence_r2, f1
