import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from keras import models, layers
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Prepare input features and binary labels
X = df.iloc[:, 1:-1].values  # Exclude ID and label columns
y = df['y'].values           # Labels

