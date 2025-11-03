import cv2
import mediapipe as mp
import numpy as np
import os
import config  # Import config file
from utils import extract_keypoints  # Import helper function

# --- Create Folders ---
for action in config.ACTIONS:
    for sequence in range(config.NO_SEQUENCES):
        try:
            os.makedirs(os.path.join(config.DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass # Folder already exists

# --- Setup MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up MediaPipe holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    print("Starting data collection...")
    
    # --- Collection Loop ---
    for action in config.ACTIONS:
        for sequence in range(config.NO_SEQUENCES):
            
            # Wait for user to be ready
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f'Press "s" to start collecting for: {action}, Video {sequence+1}/{config.NO_SEQUENCES}', 
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('s'):
                    break
            
            # "Get ready" countdown
            for t in range(5, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f'Starting in {t}', (200, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(1000)

            # --- Record Frames ---
            print(f"Recording... {action}, Video {sequence+1}")
            for frame_num in range(config.SEQUENCE_LENGTH):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                # Process image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False # Performance boost
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Show status
                cv2.putText(image, f'Recording: {action}, Video {sequence+1}, Frame {frame_num+1}', 
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                # --- Save Keypoints ---
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(config.DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    print("Data collection complete.")
    cap.release()
    cv2.destroyAllWindows()
