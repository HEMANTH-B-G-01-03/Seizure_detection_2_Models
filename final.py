import pyautogui
import cv2
import numpy as np
import tensorflow as tf  # Or torch for PyTorch models
import time
import matplotlib.pyplot as plt

# Load your pre-trained model
model = tf.keras.models.load_model('epileptic_seizure_detection_model.h5')  # Replace with your model path

