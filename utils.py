import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
from io import StringIO
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
import torch

def download_csv(url):
    """Downloads a CSV from a URL with error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return pd.DataFrame()

def preprocess_text(text):
    """Preprocesses text by tokenizing, removing stopwords, and lemmatizing."""
    if not isinstance(text, str):
        return [], ''
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens, ' '.join(tokens).strip()

def get_xanew_features(tokens, xanew_df, is_lyric=False):
    """Calculates arousal and valence from X-ANEW lexicon."""
    if xanew_df.empty:
        return 0.5, 0.5
    arousal_scores = []
    valence_scores = []
    weights = []
    for token in set(tokens):
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

def apply_pos_context(tokens, arousal, valence):
    """Adjusts scores based on POS tags and clamps to [0,1]."""
    tagged = pos_tag(tokens)
    adj_count = sum(1 for _, tag in tagged if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in tagged if tag in ['RB', 'RBR', 'RBS'])
    negation_words = {'not', 'never', 'dont', "don't"}
    negation_count = sum(1 for word, _ in tagged if word in negation_words)
    verb_neg_count = sum(1 for word, tag in tagged if tag.startswith('VB') and word in ['kill', 'destroy'])

    arousal *= (1.2 ** (adj_count / max(len(tokens), 1))) * (1.1 ** (adv_count / max(len(tokens), 1)))
    valence *= (1.2 ** (adj_count / max(len(tokens), 1))) * (1.1 ** (adv_count / max(len(tokens), 1)))
    valence *= (0.8 ** verb_neg_count)

    if negation_count % 2 == 1:
        valence = 1.0 - valence

    arousal = min(max(arousal, 0.0), 1.0)
    valence = min(max(valence, 0.0), 1.0)
    return arousal, valence

def get_bert_embeddings(texts, tokenizer, model, device, max_length=512):
    """Generates BERT embeddings with truncation for long texts."""
    embeddings = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            embeddings.append(np.zeros(768))
            continue
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length - 2:
            print(f"Warning: Truncating long text (original tokens: {len(tokens)})")
            tokens = tokens[:max_length - 2]
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

def calculate_linear_arousal(audio_df):
    weights = {'tempo': 0.4, 'loudness': 0.3, 'energy': 0.2, 'speechiness': 0.05, 'danceability': 0.05}
    arousal = sum(weights[f] * audio_df[f] for f in weights)
    return (arousal - arousal.min()) / (arousal.max() - arousal.min() + 1e-10)

def calculate_linear_valence(audio_df):
    weights = {'energy': 0.5, 'mode': 0.25, 'tempo': 0.15, 'danceability': 0.1}
    valence = sum(weights[f] * audio_df[f] for f in weights)
    return (valence - valence.min()) / (valence.max() - valence.min() + 1e-10)
