import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from io import StringIO
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import os

def setup_environment():
    drive.mount('/content/drive')
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    drive_folder = '/content/drive/MyDrive/SongEmotionPredictions/'
    os.makedirs(drive_folder, exist_ok=True)
    return drive_folder
