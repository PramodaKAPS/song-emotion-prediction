# setup.py: Environment initialization and setup
def setup_environment():
    """Sets up the Colab environment, mounts Drive, and downloads NLTK resources."""
    try:
        # Mount Google Drive with force_remount
        drive.mount('/content/drive', force_remount=True)
    except Exception as e:
        print(f"Drive mount failed: {e}. Please restart runtime and try again.")
        return None

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
