from setup import setup_environment
from data_loader import load_datasets
from preprocessing import preprocess_text, get_xanew_features, apply_pos_context
from embeddings import get_bert_embeddings
from training import train_and_evaluate
from prediction import make_predictions
from visualization import create_thayer_plot

def main():
    drive_folder = setup_environment()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    xanew_df, sentence_df, song_df = load_datasets()
    # Preprocess EmoBank
    sentence_df['tokens'], sentence_df['cleaned_text'] = zip(*sentence_df['text'].apply(preprocess_text))
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df['tokens'].apply(lambda x: get_xanew_features(x, xanew_df)))
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df.apply(lambda r: apply_pos_context(r['tokens'], r['xanew_arousal'], r['xanew_valence']), axis=1))
    sentence_embeddings = get_bert_embeddings(sentence_df['cleaned_text'], tokenizer, model, device)
    # Train models
    arousal_model, valence_model = train_and_evaluate(sentence_embeddings, sentence_df['A'].values, sentence_df['V'].values, epochs=10, drive_folder=drive_folder)
    # Preprocess Spotify
    song_df['tokens'], song_df['cleaned_lyrics'] = zip(*song_df['lyrics'].apply(preprocess_text))
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df['tokens'].apply(lambda x: get_xanew_features(x, xanew_df, is_lyric=True)))
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df.apply(lambda r: apply_pos_context(r['tokens'], r['xanew_arousal'], r['xanew_valence']), axis=1))
    lyrics_embeddings = get_bert_embeddings(song_df['cleaned_lyrics'], tokenizer, model, device)
    # Predictions
    predictions_df = make_predictions(song_df, arousal_model, valence_model, lyrics_embeddings, drive_folder)
    # Plot
    create_thayer_plot(predictions_df, drive_folder + 'thayer_plot_taylor_francis_first_1000.png')

if __name__ == '__main__':
    main()
