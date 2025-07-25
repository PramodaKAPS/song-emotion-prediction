import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')

from data_loader import load_and_prepare_data
from preprocessor import preprocess_texts, get_xanew_features, apply_pos_context, get_bert_embeddings
from model_trainer import train_and_evaluate_models
from predictor import make_predictions, combine_predictions, validate_predictions
from utils import assign_quadrant, create_thayer_plot, add_spotify_columns_to_final_csv
import pandas as pd
import numpy as np

drive_folder = './SongEmotionPredictions_FullDatasets/'
os.makedirs(drive_folder, exist_ok=True)

sentence_df, song_df, xanew_df, arousal_audio, valence_audio, audio_scaled_df = load_and_prepare_data(drive_folder)

if not sentence_df.empty:
    sentence_df['tokens'], sentence_df['cleaned_text'] = zip(*sentence_df['text'].apply(preprocess_texts))
if not song_df.empty:
    song_df['tokens'], song_df['cleaned_lyrics'] = zip(*song_df['lyrics'].apply(preprocess_texts))

if not sentence_df.empty:
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df['tokens'].apply(lambda x: get_xanew_features(x, xanew_df)))
if not song_df.empty:
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df['tokens'].apply(lambda x: get_xanew_features(x, xanew_df, is_lyric=True)))

if not sentence_df.empty:
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df.apply(
        lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))
if not song_df.empty:
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df.apply(
        lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))

sentence_embeddings = get_bert_embeddings(sentence_df['cleaned_text']) if not sentence_df.empty else np.array([])
lyrics_embeddings = get_bert_embeddings(song_df['cleaned_lyrics']) if not song_df.empty else np.array([])

arousal_model, valence_model = train_and_evaluate_models(sentence_embeddings, sentence_df, drive_folder)

arousal_text, valence_text = make_predictions(arousal_model, valence_model, lyrics_embeddings, song_df)

arousal_final, valence_final = combine_predictions(arousal_text, valence_text, arousal_audio, valence_audio)

if not song_df.empty:
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
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_taylor_francis.csv', index=False)
    print("Predictions saved locally:", drive_folder + 'song_emotion_predictions_taylor_francis.csv')

validate_predictions(song_df, valence_audio, arousal_audio, arousal_text)

if 'predictions_df' in locals() and not predictions_df.empty:
    create_thayer_plot(predictions_df, drive_folder + 'thayer_plot_taylor_francis.png')
    predictions_df['quadrant'] = predictions_df.apply(
        lambda row: assign_quadrant(row['arousal_final'], row['valence_final']), axis=1)
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis.csv', index=False)
    print("Predictions with quadrant labels saved locally:", drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis.csv')

if 'predictions_df' in locals() and not predictions_df.empty and not song_df.empty:
    add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder)
