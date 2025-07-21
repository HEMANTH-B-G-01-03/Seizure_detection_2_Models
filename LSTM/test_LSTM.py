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

# Function to generate EEG signal
def generate_eeg_signal(seizure=False):
    signal = np.zeros((n_channels, n_features))
    t = np.linspace(0, 1, n_features)  # 1 second duration
    
    for i in range(n_channels):
        # Add random oscillations from different frequency bands for non-seizure signals
        for band, (low, high) in frequency_bands.items():
            freq = random.uniform(low, high)
            signal[i] += np.sin(2 * np.pi * freq * t)  # Add basic sine wave

        if seizure:
            # Add seizure-like high-frequency activity and spike waves
            seizure_freq = 3  # 3 Hz spike-wave pattern (commonly seen in seizures)
            spike_wave_pattern = np.sin(2 * np.pi * seizure_freq * t) * 2  # Sharp waves with higher amplitude
            signal[i] += spike_wave_pattern
            
            # Add additional high-frequency (Gamma) activity for seizure-like chaotic behavior
            high_freq = random.uniform(30, 40)  # High frequency gamma activity
            signal[i] += np.sin(2 * np.pi * high_freq * t) * random.uniform(2, 5)  # Higher amplitude

        # Add noise to simulate EEG signal (controlled noise for non-seizure)
        if not seizure:
            signal[i] += np.random.normal(0, 1, n_features)  # Smaller noise for non-seizure
        else:
            signal[i] += np.random.normal(0, 2, n_features)  # More noise for seizure signals
    
    return signal.flatten()  # Flatten multi-channel data

