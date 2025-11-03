import numpy as np
import os

# --- 1. Project Parameters ---

# Path for data (Numpy arrays)
DATA_PATH = os.path.join('data')

# Model file name
MODEL_NAME = 'sign_model.h5'

# Number of sequences (videos) to collect for each action
NO_SEQUENCES = 30

# Number of frames in each sequence
SEQUENCE_LENGTH = 30

# --- 2. Keypoint Calculation (DO NOT CHANGE unless utils.py changes) ---

# This is the total number of keypoints extracted by utils.extract_keypoints()
# POSE: 33 landmarks * 3 coords = 99
# FACE: 468 landmarks * 3 coords = 1404
# LEFT HAND: 21 landmarks * 3 coords = 63
# RIGHT HAND: 21 landmarks * 3 coords = 63
# TOTAL = 99 + 1404 + 63 + 63 = 1629
KEYPOINT_COUNT = 1629

# Input shape for the LSTM model
INPUT_SHAPE = (SEQUENCE_LENGTH, KEYPOINT_COUNT)

# --- 3. Actions (This is the list you will edit) ---

# Add your 25-50 words here
# Example for 5 classes:
ACTIONS = np.array([
    'hello', 
    'thanks', 
    'iloveyou', 
    'yes', 
    'no'
])

# # Example for 25+ classes (just keep adding):
# ACTIONS = np.array([
#     'hello', 'thanks', 'iloveyou', 'yes', 'no',
#     'me', 'you', 'eat', 'drink', 'sleep',
#     'good', 'bad', 'help', 'more', 'please',
#     'home', 'work', 'school', 'outside', 'play',
#     'what', 'where', 'when', 'who', 'why'
# ])
