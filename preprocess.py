# preprocess.py: Text preprocessing and POS context application

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """Preprocesses text by tokenizing, removing stopwords, and lemmatizing."""
    if not isinstance(text, str):
        return [], ''
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens, ' '.join(tokens).strip()

def get_xanew_features(tokens, xanew_df, is_lyric=False):
    """Calculates X-ANEW based arousal and valence."""
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
    """Applies POS-based adjustments to scores."""
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
