import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from utils import assign_quadrant

def train_and_evaluate_models(sentence_embeddings, sentence_df, drive_folder):
    if not sentence_df.empty and sentence_embeddings.size > 0:
        X_text_sentence = np.hstack([sentence_embeddings, sentence_df[['xanew_arousal', 'xanew_valence']].values])
        y_arousal = sentence_df['A'].values
        y_valence = sentence_df['V'].values

        X_train, X_test, y_arousal_train, y_arousal_test = train_test_split(X_text_sentence, y_arousal, test_size=0.2, random_state=42)
        _, _, y_valence_train, y_valence_test = train_test_split(X_text_sentence, y_valence, test_size=0.2, random_state=42)

        def create_mlp_regressor(input_dim):
            try:
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(input_dim,)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                return model
            except Exception as e:
                print(f"Model creation failed: {e}")
                raise

        input_dim = X_train.shape[1]
        arousal_model = create_mlp_regressor(input_dim)
        arousal_model.fit(X_train, y_arousal_train, epochs=5, batch_size=32, verbose=1, validation_split=0.2)
        valence_model = create_mlp_regressor(input_dim)
        valence_model.fit(X_train, y_valence_train, epochs=5, batch_size=32, verbose=1, validation_split=0.2)

        y_arousal_pred = arousal_model.predict(X_test, batch_size=32).flatten()
        y_valence_pred = valence_model.predict(X_test, batch_size=32).flatten()

        arousal_mse = mean_squared_error(y_arousal_test, y_arousal_pred)
        arousal_r2 = r2_score(y_arousal_test, y_arousal_pred)
        valence_mse = mean_squared_error(y_valence_test, y_valence_pred)
        valence_r2 = r2_score(y_valence_test, y_valence_pred)

        true_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal_test, y_valence_test)]
        pred_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal_pred, y_valence_pred)]
        f1 = f1_score(true_quadrants, pred_quadrants, average='macro')
        quadrant_accuracy = accuracy_score(true_quadrants, pred_quadrants)

        print(f"Test Metrics:\nArousal MSE: {arousal_mse:.4f}, R²: {arousal_r2:.4f}\nValence MSE: {valence_mse:.4f}, R²: {valence_r2:.4f}\nF1 Score (Quadrant): {f1:.4f}\nQuadrant Accuracy: {quadrant_accuracy:.4f}")

        metrics_df = pd.DataFrame({
            'Metric': ['Arousal MSE', 'Arousal R²', 'Valence MSE', 'Valence R²', 'F1 Score (Quadrant)', 'Quadrant Accuracy'],
            'Value': [arousal_mse, arousal_r2, valence_mse, valence_r2, f1, quadrant_accuracy]
        })
        metrics_df.to_csv(drive_folder + 'test_metrics.csv', index=False)
        print("Test metrics saved locally.")

        return arousal_model, valence_model
    return None, None
