# setup.py: Environment initialization and setup

import os
import nltk
from google.colab import drive

def setup_environment():
    """Sets up the Colab environment, mounts Drive, and downloads NLTK resources."""
    # Mount Google Drive
    drive.mount('/content/drive')

    # Download NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    # Create Drive folder
    drive_folder = '/content/drive/MyDrive/SongEmotionPredictions/'
    os.makedirs(drive_folder, exist_ok=True)
    return drive_folder
