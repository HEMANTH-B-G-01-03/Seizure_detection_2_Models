import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Parameters
n_features = 256  # Number of original features (time steps per signal, e.g., 1 second @ 256 Hz sampling rate)
n_samples = 500  # Total samples to generate
seizure_proportion = 0.5  # Proportion of seizure samples
n_channels = 8  # Number of EEG channels (e.g., BioAmps EEG sensor with 8 channels)

# Frequency bands (Delta, Theta, Alpha, Beta, Gamma)
frequency_bands = {
    'Delta': (0.5, 4),   # Slow waves (sleep)
    'Theta': (4, 8),     # Light sleep and relaxation
    'Alpha': (8, 12),    # Calm alertness
    'Beta': (12, 30),    # Active thinking, alertness
    'Gamma': (30, 40)    # High-level processing, cognition
}

# Generate synthetic EEG signals
data = []
labels = []
n_seizure = int(seizure_proportion * n_samples)
n_non_seizure = n_samples - n_seizure
