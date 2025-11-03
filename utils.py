import numpy as np
import mediapipe as mp

def extract_keypoints(results):
    """
    Extracts pose, face, and hand keypoints from MediaPipe results 
    and concatenates them into a single Numpy array.
    
    If a body part is not detected, it fills its section with zeros.
    """
    # Calculate keypoints, filling with zeros if not detected
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*3)
        
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
        
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
        
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
        
    # Concatenate all keypoints
    return np.concatenate([pose, face, lh, rh])
