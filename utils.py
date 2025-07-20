# utils.py: Utilities for audio calculation, combining predictions, quadrant assignment, plotting, and merging

from sklearn.preprocessing import MinMaxScaler

def calculate_audio_scores(song_df):
    """Calculates audio-based arousal and valence."""
    audio_features = ['tempo', 'loudness', 'energy', 'speechiness', 'danceability', 'mode']
    song_df[audio_features] = song_df[audio_features].fillna(song_df[audio_features].mean())
    audio_scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(song_df[audio_features]), columns=audio_features)

    arousal_raw = (0.4 * audio_scaled_df['tempo'] + 0.3 * audio_scaled_df['loudness'] + 0.2 * audio_scaled_df['energy'] + 0.05 * audio_scaled_df['speechiness'] + 0.05 * audio_scaled_df['danceability'])
    valence_raw = (0.5 * audio_scaled_df['energy'] + 0.25 * audio_scaled_df['mode'] + 0.15 * audio_scaled_df['tempo'] + 0.1 * audio_scaled_df['danceability'])

    arousal = (arousal_raw - arousal_raw.min()) / (arousal_raw.max() - arousal_raw.min() + 1e-10)
    valence = (valence_raw - valence_raw.min()) / (valence_raw.max() - valence_raw.min() + 1e-10)
    return arousal, valence

def combine_predictions(arousal_text, valence_text, arousal_audio, valence_audio):
    """Combines text and audio predictions and scales to [-1,1]."""
    w_text, w_audio = 0.6, 0.4
    arousal_final = w_text * arousal_text + w_audio * arousal_audio if not arousal_text.empty else arousal_audio
    valence_final = w_text * valence_text + w_audio * valence_audio if not valence_text.empty else valence_audio
    arousal_final = 2 * arousal_final - 1
    valence_final = 2 * valence_final - 1
    return arousal_final, valence_final

def assign_quadrant(arousal, valence):
    """Assigns emotional quadrant."""
    if arousal >= 0 and valence >= 0:
        return 'Happy/Excited'
    elif arousal >= 0 and valence < 0:
        return 'Angry/Stressed'
    elif arousal < 0 and valence >= 0:
        return 'Calm/Peaceful'
    else:
        return 'Sad/Depressed'

def create_thayer_plot(predictions_df, output_file):
    """Creates Thayer's plot."""
    if predictions_df.empty:
        print("No data for plot.")
        return
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=predictions_df, x='valence_final', y='arousal_final',
                    hue='valence_final', size='arousal_final',
                    palette='viridis', alpha=0.6, sizes=(20, 200))
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.text(-0.5, 0.5, 'Angry/Stressed', fontsize=10, ha='center')
    plt.text(-0.5, -0.5, 'Sad/Depressed', fontsize=10, ha='center')
    plt.text(0.5, 0.5, 'Happy/Excited', fontsize=10, ha='center')
    plt.text(0.5, -0.5, 'Calm/Peaceful', fontsize=10, ha='center')
    top_songs = predictions_df.nlargest(5, 'arousal_final')
    for _, row in top_songs.iterrows():
        plt.text(row['valence_final'], row['arousal_final'], row['track_name'],
                 fontsize=8, ha='right', va='bottom')
    plt.xlabel('Valence (Negative to Positive)', fontsize=12)
    plt.ylabel('Arousal (Calm to Excited)', fontsize=12)
    plt.title('Thayer\'s Emotion Plane for Spotify Songs (Taylor & Francis 2020)', fontsize=14)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Thayer's plot saved to {output_file}")

def add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder):
    """Merges predictions into original song_df and saves."""
    if predictions_df.empty or song_df.empty:
        print("No data to merge.")
        return
    final_df = song_df.copy()
    final_df = final_df.join(predictions_df.set_index(['track_id', 'track_name', 'track_artist']), on=['track_id', 'track_name', 'track_artist'])
    final_output_file = drive_folder + 'spotify_full_with_predictions.csv'
    final_df.to_csv(final_output_file, index=False)
    print(f"Final output CSV with all Spotify columns and predictions saved to: {final_output_file}")
