def create_mlp_regressor(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(X, y_a, y_v, epochs=10, drive_folder=None):
    arousal_model = create_mlp_regressor(X.shape[1])
    arousal_history = arousal_model.fit(X, y_a, epochs=epochs, verbose=1, validation_split=0.2)

    valence_model = create_mlp_regressor(X.shape[1])
    valence_history = valence_model.fit(X, y_v, epochs=epochs, verbose=1, validation_split=0.2)

    y_a_pred = arousal_model.predict(X).flatten()
    y_v_pred = valence_model.predict(X).flatten()

    a_mse = mean_squared_error(y_a, y_a_pred)
    a_r2 = r2_score(y_a, y_a_pred)
    v_mse = mean_squared_error(y_v, y_v_pred)
    v_r2 = r2_score(y_v, y_v_pred)

    true_quadrants = [assign_quadrant(a, v) for a, v in zip(y_a, y_v)]
    pred_quadrants = [assign_quadrant(a, v) for a, v in zip(y_a_pred, y_v_pred)]
    f1 = f1_score(true_quadrants, pred_quadrants, average='macro')

    print(f"Training Metrics:\nArousal MSE: {a_mse:.4f}, R²: {a_r2:.4f}\nValence MSE: {v_mse:.4f}, R²: {v_r2:.4f}\nF1 Score (Quadrant): {f1:.4f}")

    metrics_df = pd.DataFrame({
        'Metric': ['Arousal MSE', 'Arousal R²', 'Valence MSE', 'Valence R²', 'F1 Score (Quadrant)'],
        'Value': [a_mse, a_r2, v_mse, v_r2, f1]
    })
    metrics_df.to_csv(drive_folder + 'training_metrics_first_1000.csv', index=False)
    print("Training metrics saved to Google Drive.")
    
    return arousal_model, valence_model
