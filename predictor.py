import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def make_predictions(arousal_model, valence_model, lyrics_embeddings, song_df):
    if not song_df.empty and lyrics_embeddings.size > 0:
        X_text_lyrics = np.hstack([lyrics_embeddings, song_df[['xanew_arousal', 'xanew_valence']].values])
        arousal_text = pd.Series(arousal_model.predict(X_text_lyrics, batch_size=32).flatten())
        valence_text = pd.Series(valence_model.predict(X_text_lyrics, batch_size=32).flatten())
    else:
        arousal_text = pd.Series()
        valence_text = pd.Series()
    return arousal_text, valence_text

def combine_predictions(arousal_text, valence_text, arousal_audio, valence_audio):
    w_text = 0.6
    w_audio = 0.4
    arousal_final = w_text * arousal_text + w_audio * arousal_audio if not arousal_text.empty else arousal_audio
    valence_final = w_text * valence_text + w_audio * valence_audio if not valence_text.empty else valence_audio

    arousal_final = 2 * arousal_final - 1
    valence_final = 2 * valence_final - 1
    return arousal_final, valence_final

def validate_predictions(song_df, valence_audio, arousal_audio, arousal_text):
    if not song_df.empty and not valence_audio.empty:
        mse_valence = mean_squared_error(song_df['valence'], valence_audio)
        print(f"Valence MSE (linear audio vs. Spotify valence): {mse_valence:.4f}")

    if not arousal_audio.empty and not arousal_text.empty:
        corr, _ = pearsonr(arousal_audio, arousal_text)
        print(f"Arousal correlation (audio vs. text): {corr:.4f}")
