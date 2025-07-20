def download_csv(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return pd.DataFrame()

def load_datasets(limit=1000):
    xanew_df = download_csv('https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv')
    if not xanew_df.empty:
        xanew_df = xanew_df[['Word', 'V.Mean.Sum', 'A.Mean.Sum']].rename(columns={'Word': 'word', 'V.Mean.Sum': 'valence', 'A.Mean.Sum': 'arousal'})
        xanew_df[['valence', 'arousal']] = MinMaxScaler().fit_transform(xanew_df[['valence', 'arousal']])
    
    sentence_df = download_csv('https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/emobank.csv')
    song_df = download_csv('https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/spotify_songs.csv')
    
    if not sentence_df.empty:
        sentence_df = sentence_df[sentence_df['split'] == 'train'].head(limit)
    if not song_df.empty:
        song_df = song_df.head(limit)
    
    return xanew_df, sentence_df, song_df

