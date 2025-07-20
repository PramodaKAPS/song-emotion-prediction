# data_loader.py: Functions for downloading and loading datasets

import requests
from io import StringIO
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def download_csv(url):
    """Downloads a CSV from a URL with error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return pd.DataFrame()

def load_datasets():
    """Loads and preprocesses all datasets."""
    XANEW_URL = 'https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv'
    EMOBANK_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/emobank.csv'
    SPOTIFY_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/spotify_songs.csv'

    xanew_df = download_csv(XANEW_URL)
    if not xanew_df.empty:
        xanew_df = xanew_df[['Word', 'V.Mean.Sum', 'A.Mean.Sum']].rename(columns={'Word': 'word', 'V.Mean.Sum': 'valence', 'A.Mean.Sum': 'arousal'})
        xanew_scaler = MinMaxScaler()
        xanew_df[['valence', 'arousal']] = xanew_scaler.fit_transform(xanew_df[['valence', 'arousal']])

    sentence_df = download_csv(EMOBANK_URL)
    song_df = download_csv(SPOTIFY_URL)

    # Subset for testing (first 50 rows) - comment out for full datasets
    if not sentence_df.empty:
        sentence_df = sentence_df[sentence_df['split'] == 'train'].head(50)
    if not song_df.empty:
        song_df = song_df.head(50)

    # Normalize EmoBank arousal/valence
    if not sentence_df.empty:
        emo_scaler = MinMaxScaler()
        sentence_df[['V', 'A']] = emo_scaler.fit_transform(sentence_df[['V', 'A']])

    return xanew_df, sentence_df, song_df
