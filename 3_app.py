import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model
import pyttsx3
import config  # Import config file
from utils import extract_keypoints  # Import helper function

# --- 1. Load Model and Setup TTS ---
try:
    model = load_model(config.MODEL_NAME)
    print(f"Model '{config.MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Make sure '{config.MODEL_NAME}' exists. Run 2_train_model.py first.")
    exit()

engine = pyttsx3.init()

# --- 2. Setup MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- 3. Real-Time Detection Variables ---
sequence = []
predictions = []
threshold = 0.90         # Confidence threshold
last_spoken_word = None
speak_cooldown = 0
COOLDOWN_FRAMES = 30     # Wait 30 frames (approx. 1 sec) before speaking again

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up MediaPipe holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Process image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- 5. Prediction Logic ---
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-config.SEQUENCE_LENGTH:] # Keep list at 30 frames

        current_prediction = "" # The word to display

        if len(sequence) == config.SEQUENCE_LENGTH:
            # Predict
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_action = config.ACTIONS[np.argmax(res)]
            confidence = res[np.argmax(res)]
            
            # --- 6. Spam Control & TTS ---
            if confidence > threshold:
                current_prediction = f'{predicted_action} ({confidence*100:.1f}%)'
                
                # Check if this is a new word and cooldown is over
                if predicted_action != last_spoken_word and speak_cooldown == 0:
                    print(f"Speaking: {predicted_action}")
                    engine.say(predicted_action)
                    engine.runAndWait()
                    
                    last_spoken_word = predicted_action
                    speak_cooldown = COOLDOWN_FRAMES # Start cooldown
            else:
                last_spoken_word = None

        # Decrement cooldown timer
        if speak_cooldown > 0:
            speak_cooldown -= 1
            
        # --- 7. Visualization ---
        cv2.putText(image, current_prediction, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if speak_cooldown > 0:
            cv2.putText(image, "COOLDOWN", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
