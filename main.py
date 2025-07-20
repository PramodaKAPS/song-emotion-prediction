# main.py: Main script to run the full pipeline

# Added missing imports
from transformers import AutoTokenizer, AutoModel
from google.colab import drive
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import os

from setup import setup_environment
from data_loader import download_csv  # Assuming data_loader.py has load_datasets function; adjust if needed
from preprocess import preprocess_text, get_xanew_features, apply_pos_context
from features import get_bert_embeddings
from model import create_mlp_regressor, compute_metrics  # Adjusted to match your model.py
from utils import calculate_audio_scores, combine_predictions, assign_quadrant, create_thayer_plot, add_spotify_columns_to_final_csv

def main():
    drive_folder = setup_environment()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    xanew_df = download_csv(XANEW_URL)
    sentence_df = download_csv(EMOBANK_URL)
    song_df = download_csv(SPOTIFY_URL)
    
    # Preprocess and extract features for EmoBank
    sentence_df['tokens'], sentence_df['cleaned_text'] = zip(*sentence_df['text'].apply(preprocess_text))
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df['tokens'].apply(lambda x: get_xanew_features(x)))
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df.apply(lambda r: apply_pos_context(r['tokens'], r['xanew_arousal'], r['xanew_valence']), axis=1))
    sentence_embeddings = get_bert_embeddings(sentence_df['cleaned_text'], tokenizer, model, device)
    
    # Train models
    X_train = np.hstack([sentence_embeddings, sentence_df[['xanew_arousal', 'xanew_valence']].values])
    y_arousal, y_valence = sentence_df['A'].values, sentence_df['V'].values
    arousal_model = create_mlp_regressor(X_train.shape[1])
    arousal_model.fit(X_train, y_arousal, epochs=10, verbose=1, validation_split=0.2)
    valence_model = create_mlp_regressor(X_train.shape[1])
    valence_model.fit(X_train, y_valence, epochs=10, verbose=1, validation_split=0.2)
    
    # Compute metrics
    y_a_pred = arousal_model.predict(X_train).flatten()
    y_v_pred = valence_model.predict(X_train).flatten()
    arousal_mse, arousal_r2, valence_mse, valence_r2, f1 = compute_metrics(y_arousal, y_a_pred, y_valence, y_v_pred)
    
    # (Rest of the main function remains the same as your original)

if __name__ == '__main__':
    main()


