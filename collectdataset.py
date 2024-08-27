import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe hands module for hand detection and landmark estimation
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Parameters
image_size = 128  # Resize captured image size
num_images = 100  # Number of images to capture for each gesture
gestures = [ "hello","help", "please","yes"]  # List of gestures

# Create directories if they don't exist
data_dir = "dataset"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
for gesture in gestures:
    gesture_dir = os.path.join(data_dir, gesture)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

# Function to capture images of the hand gesture
def capture_hand_images(gesture):
    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Starting to capture images for {gesture}. Press 's' to start capturing and 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference
        image_rgb.flags.writeable = False

        # Perform hand detection and landmark estimation
        results = hands.process(image_rgb)

        # Check if hand(s) detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks (skeleton) on the image
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Initialize min and max values for bounding box
                x_min, y_min = frame.shape[1], frame.shape[0]
                x_max, y_max = 0, 0

                # Iterate through all landmarks to find bounding box
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y

                # Crop hand region based on detected bounding box
                hand_crop = frame[y_min:y_max, x_min:x_max]

                # Resize cropped hand region
                hand_resized = cv2.resize(hand_crop, (image_size, image_size))

                # Save the resized hand image to the dataset directory
                image_path = os.path.join(data_dir, gesture, f'{count}.jpg')
                cv2.imwrite(image_path, hand_resized)
                print(f"Captured image {count} for {gesture}")
                count += 1

                # Stop capturing after num_images
                if count >= num_images:
                    break

        # Display the frame
        cv2.imshow('Hand Gesture Capture', frame)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Capture images for each gesture
for gesture in gestures:
    capture_hand_images(gesture)