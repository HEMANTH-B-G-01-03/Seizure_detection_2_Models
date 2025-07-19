import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model = load_model('epileptic_seizure_detection_model.h5')

# Load the saved test set
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Normalize input data
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
X_test = np.nan_to_num(X_test)
