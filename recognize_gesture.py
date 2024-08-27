import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Initialize MediaPipe hands module for hand detection and landmark estimation
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = tf.keras.models.load_model('gesture_model.h5')

# Parameters
image_size = 64
class_names = ["hello","help", "please","yes"]  # Update this list with your gesture names

# Function to preprocess frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size))  # Ensure image size is (64, 64)
    reshaped = resized.reshape(1, image_size, image_size, 1)  # Reshape to match (1, 64, 64, 1)
    return reshaped / 255.0  # Normalize pixel values to [0, 1]

def draw_hand_landmarks(frame, landmarks):
    if landmarks:
        # Draw connections between nodes (landmarks)
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_index = connection[0]
            end_index = connection[1]
            
            # Extract start and end landmarks
            start_landmark = landmarks.landmark[start_index]
            end_landmark = landmarks.landmark[end_index]
            
            # Calculate pixel coordinates
            start_point = (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0]))
            end_point = (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0]))
            
            # Draw line
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)  # White color, thickness 2
        
        # Draw landmarks (nodes) as circles
        for landmark in landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red color, filled circle

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Hand pose estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks)

            # Extract bounding box of the hand region
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Ensure bounding box coordinates are within frame dimensions
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # Crop hand region
            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size > 0:
                try:
                    preprocessed_frame = preprocess_frame(hand_crop)

                    # Predict the gesture
                    predictions = model.predict(preprocessed_frame)
                    gesture = class_names[np.argmax(predictions)]

                    # Display the gesture
                    cv2.putText(frame, f'Gesture: {gesture}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error during prediction: {e}")

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()