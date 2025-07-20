# main.py: Main script to run the full pipeline

from setup import setup_environment
from data_loader import download_csv, load_datasets  # Assuming data_loader.py has load_datasets function; adjust if needed
from preprocess import preprocess_text, get_xanew_features, apply_pos_context
from features import get_bert_embeddings
from model import train_and_evaluate
from utils import calculate_audio_scores, combine_predictions, assign_quadrant, create_thayer_plot, add_spotify_columns_to_final_csv

def main():
    drive_folder = setup_environment()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    xanew_df, sentence_df, song_df = load_datasets()
    
    # Preprocess and extract features for EmoBank
    sentence_df['tokens'], sentence_df['cleaned_text'] = zip(*sentence_df['text'].apply(preprocess_text))
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df['tokens'].apply(lambda x: get_xanew_features(x)))
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df.apply(lambda r: apply_pos_context(r['tokens'], r['xanew_arousal'], r['xanew_valence']), axis=1))
    sentence_embeddings = get_bert_embeddings(sentence_df['cleaned_text'], tokenizer, model, device)
    
    # Train models
    X_train = np.hstack([sentence_embeddings, sentence_df[['xanew_arousal', 'xanew_valence']].values])
    y_arousal, y_valence = sentence_df['A'].values, sentence_df['V'].values
    arousal_model, valence_model, arousal_mse, arousal_r2, valence_mse, valence_r2, f1 = train_and_evaluate(X_train, y_arousal, y_valence)
    
    # Preprocess and extract features for Spotify
    song_df['tokens'], song_df['cleaned_lyrics'] = zip(*song_df['lyrics'].apply(preprocess_text))
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df['tokens'].apply(lambda x: get_xanew_features(x, is_lyric=True)))
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df.apply(lambda r: apply_pos_context(r['tokens'], r['xanew_arousal'], r['xanew_valence']), axis=1))
    lyrics_embeddings = get_bert_embeddings(song_df['cleaned_lyrics'], tokenizer, model, device)
    
    # Predict
    X_predict = np.hstack([lyrics_embeddings, song_df[['xanew_arousal', 'xanew_valence']].values])
    arousal_text = pd.Series(arousal_model.predict(X_predict).flatten())
    valence_text = pd.Series(valence_model.predict(X_predict).flatten())
    
    # Audio scores
    arousal_audio, valence_audio = calculate_audio_scores(song_df)
    
    # Combine and scale
    arousal_final, valence_final = combine_predictions(arousal_text, valence_text, arousal_audio, valence_audio)
    
    # Create predictions_df
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
    
    # Validation (unchanged)
    if not song_df.empty and not valence_audio.empty:
        mse_valence = mean_squared_error(song_df['valence'], valence_audio)
        print(f"Valence MSE (linear audio vs. Spotify valence): {mse_valence:.4f}")

    if not arousal_audio.empty and not arousal_text.empty:
        corr, _ = pearsonr(arousal_audio, arousal_text)
        print(f"Arousal correlation (audio vs. text): {corr:.4f}")

    # Plot
    create_thayer_plot(predictions_df, drive_folder + 'thayer_plot_taylor_francis.png')
    
    # Quadrant
    predictions_df['quadrant'] = predictions_df.apply(lambda row: assign_quadrant(row['arousal_final'], row['valence_final']), axis=1)
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis.csv', index=False)
    
    # Merge and save final CSV with all Spotify columns
    add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder)

if __name__ == '__main__':
    main()
