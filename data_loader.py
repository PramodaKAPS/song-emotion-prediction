import pandas as pd
from io import StringIO
import requests
from sklearn.preprocessing import MinMaxScaler
import os
import nltk

def load_and_prepare_data(drive_folder):
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    XANEW_URL = 'https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv'
    EMOBANK_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/emobank.csv'
    SPOTIFY_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/spotify_songs.csv'

    def download_csv(url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except requests.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return pd.DataFrame()

    xanew_df = download_csv(XANEW_URL)
    if not xanew_df.empty:
        xanew_df = xanew_df[['Word', 'V.Mean.Sum', 'A.Mean.Sum']].rename(columns={'Word': 'word', 'V.Mean.Sum': 'valence', 'A.Mean.Sum': 'arousal'})
        xanew_scaler = MinMaxScaler()
        xanew_df[['valence', 'arousal']] = xanew_scaler.fit_transform(xanew_df[['valence', 'arousal']])

    sentence_df = download_csv(EMOBANK_URL)
    song_df = download_csv(SPOTIFY_URL)

    if not sentence_df.empty:
        sentence_df = sentence_df[sentence_df['split'] == 'train']
        print(f"Emobank rows: {len(sentence_df)}")

    if not song_df.empty:
        print(f"Full Spotify rows for predictions: {len(song_df)}")

    if not sentence_df.empty:
        emo_scaler = MinMaxScaler()
        sentence_df[['V', 'A']] = emo_scaler.fit_transform(sentence_df[['V', 'A']])

    audio_features = ['tempo', 'loudness', 'energy', 'speechiness', 'danceability', 'mode']

    if not song_df.empty:
        song_df[audio_features] = song_df[audio_features].fillna(song_df[audio_features].mean())
        scaler_audio = MinMaxScaler()
        audio_scaled = scaler_audio.fit_transform(song_df[audio_features])
        audio_scaled_df = pd.DataFrame(audio_scaled, columns=audio_features)
    else:
        audio_scaled_df = pd.DataFrame()

    arousal_audio = calculate_linear_arousal(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()
    valence_audio = calculate_linear_valence(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()

    return sentence_df, song_df, xanew_df, arousal_audio, valence_audio, audio_scaled_df

def calculate_linear_arousal(audio_df):
    weights = {'tempo': 0.4, 'loudness': 0.3, 'energy': 0.2, 'speechiness': 0.05, 'danceability': 0.05}
    arousal = sum(weights[f] * audio_df[f] for f in weights)
    arousal = (arousal - arousal.min()) / (arousal.max() - arousal.min() + 1e-10)
    return arousal

def calculate_linear_valence(audio_df):
    weights = {'energy': 0.5, 'mode': 0.25, 'tempo': 0.15, 'danceability': 0.1}
    valence = sum(weights[f] * audio_df[f] for f in weights)
    valence = (valence - valence.min()) / (valence.max() - valence.min() + 1e-10)
    return valence
