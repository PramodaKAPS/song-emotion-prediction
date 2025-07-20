import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os  # Added for directory creation
import tensorflow as tf  # Added for neural network with epochs
from sklearn.metrics import r2_score, f1_score  # Added for R² and F1

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For headless plotting on server
import matplotlib
matplotlib.use('Agg')

# Create local folder if it doesn't exist
drive_folder = './SongEmotionPredictions/'
os.makedirs(drive_folder, exist_ok=True)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Load DistilBERT once
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# URLs
XANEW_URL = 'https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv'
EMOBANK_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/emobank.csv'
SPOTIFY_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/spotify_songs.csv'

# Download datasets with error handling
def download_csv(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

# Load datasets
xanew_df = download_csv(XANEW_URL)
if not xanew_df.empty:
    xanew_df = xanew_df[['Word', 'V.Mean.Sum', 'A.Mean.Sum']].rename(columns={'Word': 'word', 'V.Mean.Sum': 'valence', 'A.Mean.Sum': 'arousal'})
    xanew_scaler = MinMaxScaler()
    xanew_df[['valence', 'arousal']] = xanew_scaler.fit_transform(xanew_df[['valence', 'arousal']])

sentence_df = download_csv(EMOBANK_URL)
song_df = download_csv(SPOTIFY_URL)

# Subset for testing (first 5000 rows for training and predictions)
if not sentence_df.empty:
    sentence_df = sentence_df[sentence_df['split'] == 'train'].iloc[0:5000]  # First 5000 rows for training
if not song_df.empty:
    song_df = song_df.head(5000)  # First 5000 rows for predictions (updated as requested)

# Normalize EmoBank arousal/valence
if not sentence_df.empty:
    emo_scaler = MinMaxScaler()
    sentence_df[['V', 'A']] = emo_scaler.fit_transform(sentence_df[['V', 'A']])

# Audio features
audio_features = ['tempo', 'loudness', 'energy', 'speechiness', 'danceability', 'mode']

if not song_df.empty:
    # Handle missing values
    song_df[audio_features] = song_df[audio_features].fillna(song_df[audio_features].mean())

    # Normalize audio features
    scaler_audio = MinMaxScaler()
    audio_scaled = scaler_audio.fit_transform(song_df[audio_features])
    audio_scaled_df = pd.DataFrame(audio_scaled, columns=audio_features)

# Linear regression coefficients (Taylor & Francis 2020, fixed weights)
def calculate_linear_arousal(audio_df):
    weights = {
        'tempo': 0.4,
        'loudness': 0.3,
        'energy': 0.2,
        'speechiness': 0.05,
        'danceability': 0.05
    }
    arousal = sum(weights[f] * audio_df[f] for f in weights)
    arousal = (arousal - arousal.min()) / (arousal.max() - arousal.min() + 1e-10)
    return arousal

def calculate_linear_valence(audio_df):
    # Removed 'energy_extra'; redistributed weights
    weights = {
        'energy': 0.5,  # Increased to cover energy_extra
        'mode': 0.25,
        'tempo': 0.15,
        'danceability': 0.1
    }
    valence = sum(weights[f] * audio_df[f] for f in weights)
    valence = (valence - valence.min()) / (valence.max() - valence.min() + 1e-10)
    return valence

arousal_audio = calculate_linear_arousal(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()
valence_audio = calculate_linear_valence(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return [], ''
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens, ' '.join(tokens).strip()

# X-ANEW features
def get_xanew_features(tokens, is_lyric=False):
    if xanew_df.empty:
        return 0.5, 0.5
    arousal_scores = []
    valence_scores = []
    weights = []
    for token in set(tokens):  # Use set to avoid redundancy
        if token in xanew_df['word'].values:
            row = xanew_df[xanew_df['word'] == token]
            count = tokens.count(token)
            arousal_scores.append(row['arousal'].values[0] * count)
            valence_scores.append(row['valence'].values[0] * count)
            weight = 2.0 if is_lyric and count > 1 else 1.0
            weights.append(weight * count)
    total_weight = sum(weights) if weights else 1.0
    arousal = sum(arousal_scores) / total_weight if arousal_scores else 0.5
    valence = sum(valence_scores) / total_weight if valence_scores else 0.5
    return arousal, valence

# POS tagging with single adjustment per sentence and clamping
def apply_pos_context(tokens, arousal, valence):
    tagged = pos_tag(tokens)
    adj_count = sum(1 for _, tag in tagged if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in tagged if tag in ['RB', 'RBR', 'RBS'])
    negation_words = {'not', 'never', 'dont', "don't"}
    negation_count = sum(1 for word, _ in tagged if word in negation_words)
    verb_neg_count = sum(1 for word, tag in tagged if tag.startswith('VB') and word in ['kill', 'destroy'])

    # Apply adjustments once
    arousal *= (1.2 ** (adj_count / max(len(tokens), 1))) * (1.1 ** (adv_count / max(len(tokens), 1)))
    valence *= (1.2 ** (adj_count / max(len(tokens), 1))) * (1.1 ** (adv_count / max(len(tokens), 1)))
    valence *= (0.8 ** verb_neg_count)

    # Flip valence if odd negations
    if negation_count % 2 == 1:
        valence = 1.0 - valence

    # Clamp to [0, 1]
    arousal = min(max(arousal, 0.0), 1.0)
    valence = min(max(valence, 0.0), 1.0)
    return arousal, valence

# Preprocess texts
if not sentence_df.empty:
    sentence_df['tokens'], sentence_df['cleaned_text'] = zip(*sentence_df['text'].apply(preprocess_text))
if not song_df.empty:
    song_df['tokens'], song_df['cleaned_lyrics'] = zip(*song_df['lyrics'].apply(preprocess_text))

# Get X-ANEW features
if not sentence_df.empty:
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df['tokens'].apply(lambda x: get_xanew_features(x)))
if not song_df.empty:
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df['tokens'].apply(lambda x: get_xanew_features(x, is_lyric=True)))

# Apply POS context
if not sentence_df.empty:
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df.apply(
        lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))
if not song_df.empty:
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df.apply(
        lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))

# DistilBERT embeddings with improved token-based splitting and truncation
def get_bert_embeddings(texts, max_length=512):
    embeddings = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            embeddings.append(np.zeros(768))
            continue
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length - 2:
            print(f"Warning: Truncating long text (original tokens: {len(tokens)})")
            tokens = tokens[:max_length - 2]  # Explicit truncation
        chunks = [tokens[i:i + (max_length - 2)] for i in range(0, len(tokens), max_length - 2)]
        sentence_embeddings = []
        for chunk in chunks:
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            sentence_embeddings.append(embedding)
        avg_embedding = np.mean(sentence_embeddings, axis=0) if sentence_embeddings else np.zeros(768)
        embeddings.append(avg_embedding)
    return np.array(embeddings)

# Extract embeddings
sentence_embeddings = get_bert_embeddings(sentence_df['cleaned_text']) if not sentence_df.empty else np.array([])
lyrics_embeddings = get_bert_embeddings(song_df['cleaned_lyrics']) if not song_df.empty else np.array([])

# Quadrant labels (adjusted for [-1,1] scale) - Moved earlier for F1 calculation
def assign_quadrant(arousal, valence):
    if arousal >= 0 and valence >= 0:
        return 'Happy/Excited'
    elif arousal >= 0 and valence < 0:
        return 'Angry/Stressed'
    elif arousal < 0 and valence >= 0:
        return 'Calm/Peaceful'
    else:
        return 'Sad/Depressed'

# Lyrics-based training with neural network for epochs
if not sentence_df.empty and sentence_embeddings.size > 0:
    X_text_sentence = np.hstack([sentence_embeddings, sentence_df[['xanew_arousal', 'xanew_valence']].values])
    y_arousal = sentence_df['A'].values
    y_valence = sentence_df['V'].values

    # Define simple MLP regressor model (updated with Input layer)
    def create_mlp_regressor(input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Regression output
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # Train arousal model with epochs
    input_dim = X_text_sentence.shape[1]
    arousal_model = create_mlp_regressor(input_dim)
    arousal_history = arousal_model.fit(X_text_sentence, y_arousal, epochs=10, verbose=1, validation_split=0.2)

    # Train valence model with epochs
    valence_model = create_mlp_regressor(input_dim)
    valence_history = valence_model.fit(X_text_sentence, y_valence, epochs=10, verbose=1, validation_split=0.2)

    # Predict on training data for metrics
    y_arousal_pred = arousal_model.predict(X_text_sentence).flatten()
    y_valence_pred = valence_model.predict(X_text_sentence).flatten()

    # Compute training "accuracy" (MSE and R² as proxies)
    arousal_mse = mean_squared_error(y_arousal, y_arousal_pred)
    arousal_r2 = r2_score(y_arousal, y_arousal_pred)
    valence_mse = mean_squared_error(y_valence, y_valence_pred)
    valence_r2 = r2_score(y_valence, y_valence_pred)

    # Discretize to quadrants for F1 score
    true_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal, y_valence)]
    pred_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal_pred, y_valence_pred)]
    f1 = f1_score(true_quadrants, pred_quadrants, average='macro')

    # Print metrics
    print(f"Training Metrics:\nArousal MSE: {arousal_mse:.4f}, R²: {arousal_r2:.4f}\nValence MSE: {valence_mse:.4f}, R²: {valence_r2:.4f}\nF1 Score (Quadrant): {f1:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Arousal MSE', 'Arousal R²', 'Valence MSE', 'Valence R²', 'F1 Score (Quadrant)'],
        'Value': [arousal_mse, arousal_r2, valence_mse, valence_r2, f1]
    })
    metrics_df.to_csv(drive_folder + 'training_metrics.csv', index=False)
    print("Training metrics saved locally.")

    # Predict on Spotify data
    if not song_df.empty and lyrics_embeddings.size > 0:
        X_text_lyrics = np.hstack([lyrics_embeddings, song_df[['xanew_arousal', 'xanew_valence']].values])
        arousal_text = pd.Series(arousal_model.predict(X_text_lyrics).flatten())
        valence_text = pd.Series(valence_model.predict(X_text_lyrics).flatten())
    else:
        arousal_text = pd.Series()
        valence_text = pd.Series()
else:
    arousal_text = pd.Series()
    valence_text = pd.Series()

# Combine predictions
w_text = 0.6
w_audio = 0.4
arousal_final = w_text * arousal_text + w_audio * arousal_audio if not arousal_text.empty else arousal_audio
valence_final = w_text * valence_text + w_audio * valence_audio if not valence_text.empty else valence_audio

# Scale final arousal and valence to [-1, 1]
arousal_final = 2 * arousal_final - 1
valence_final = 2 * valence_final - 1

# Save predictions locally
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
    print("Predictions saved locally: " + drive_folder + 'song_emotion_predictions_taylor_francis.csv')

# Validation
if not song_df.empty and not valence_audio.empty:
    mse_valence = mean_squared_error(song_df['valence'], valence_audio)
    print(f"Valence MSE (linear audio vs. Spotify valence): {mse_valence:.4f}")

if not arousal_audio.empty and not arousal_text.empty:
    corr, _ = pearsonr(arousal_audio, arousal_text)
    print(f"Arousal correlation (audio vs. text): {corr:.4f}")

# Thayer's Plot (adjusted for [-1,1] scale)
def create_thayer_plot(predictions_df, output_file=drive_folder + 'thayer_plot_taylor_francis.png'):
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
    print(f"Thayer's plot saved locally: '{output_file}'")

if 'predictions_df' in locals() and not predictions_df.empty:
    create_thayer_plot(predictions_df)

# Quadrant labels (adjusted for [-1,1] scale)
def assign_quadrant(arousal, valence):
    if arousal >= 0 and valence >= 0:
        return 'Happy/Excited'
    elif arousal >= 0 and valence < 0:
        return 'Angry/Stressed'
    elif arousal < 0 and valence >= 0:
        return 'Calm/Peaceful'
    else:
        return 'Sad/Depressed'

if 'predictions_df' in locals() and not predictions_df.empty:
    predictions_df['quadrant'] = predictions_df.apply(
        lambda row: assign_quadrant(row['arousal_final'], row['valence_final']), axis=1)
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis.csv', index=False)
    print("Predictions with quadrant labels saved locally: " + drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis.csv')

# Function to add Spotify columns to final output CSV (as requested)
def add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder):
    if predictions_df.empty or song_df.empty:
        print("No data to merge.")
        return

    # Merge predictions into the original song_df (adds all Spotify columns + predictions)
    final_df = song_df.copy()  # Copy original Spotify DataFrame
    final_df = final_df.join(predictions_df.set_index(['track_id', 'track_name', 'track_artist']), on=['track_id', 'track_name', 'track_artist'])

    # Save the final enriched DataFrame
    final_output_file = drive_folder + 'spotify_full_with_predictions.csv'
    final_df.to_csv(final_output_file, index=False)
    print(f"Final output CSV with all Spotify columns and predictions saved locally: {final_output_file}")

# Call the function after predictions are ready
if 'predictions_df' in locals() and not predictions_df.empty and not song_df.empty:
    add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder)
