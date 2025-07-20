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

def make_predictions(song_df, arousal_model, valence_model, lyrics_embeddings, drive_folder):
    audio_features = ['tempo', 'loudness', 'energy', 'speechiness', 'danceability', 'mode']
    song_df[audio_features] = song_df[audio_features].fillna(song_df[audio_features].mean())
    audio_scaled = MinMaxScaler().fit_transform(song_df[audio_features])
    audio_scaled_df = pd.DataFrame(audio_scaled, columns=audio_features)

    arousal_audio = calculate_linear_arousal(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()
    valence_audio = calculate_linear_valence(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()

    if not song_df.empty and lyrics_embeddings.size > 0:
        X_text_lyrics = np.hstack([lyrics_embeddings, song_df[['xanew_arousal', 'xanew_valence']].values])
        arousal_text = pd.Series(arousal_model.predict(X_text_lyrics).flatten())
        valence_text = pd.Series(valence_model.predict(X_text_lyrics).flatten())
    else:
        arousal_text = pd.Series()
        valence_text = pd.Series()

    w_text = 0.6
    w_audio = 0.4
    arousal_final = w_text * arousal_text + w_audio * arousal_audio if not arousal_text.empty else arousal_audio
    valence_final = w_text * valence_text + w_audio * valence_audio if not valence_text.empty else valence_audio

    arousal_final = 2 * arousal_final - 1
    valence_final = 2 * valence_final - 1

    predictions_df = pd.DataFrame({
        'track_id': song_df['track_id'],
        'track_name': song_df['track_name'],
        'track_artist': song_df['track_artist'],
        'arousal_text': arousal_text,
        'valence_text': valence_text,
        'arousal_audio': arousal_audio,
        'valence_audio': valence_audio,
        'arousal_final': arousal_final,
        'valence_final': valence_final
    })
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_taylor_francis_first_1000.csv', index=False)
    print("Predictions saved to Google Drive: " + drive_folder + 'song_emotion_predictions_taylor_francis_first_1000.csv')
    
    return predictions_df
