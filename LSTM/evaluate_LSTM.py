import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model = load_model('epileptic_seizure_detection_lstm_model.h5')

# Load the saved test set
X_test = np.load('X_test1.npy')
y_test = np.load('y_test1.npy')

# Reshape X_test to 3D for LSTM
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

