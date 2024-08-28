# Sign Language Recognition using OpenCV

## Overview

This project implements a sign language recognition system using OpenCV. The system is designed to recognize and translate sign language gestures into text, facilitating communication for individuals who use sign language.

## Features

- Real-time gesture recognition using a webcam
- Accurate translation of sign language gestures to text
- Built-in support for various sign language alphabets (e.g., American Sign Language)

## Requirements

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Keras
- Mediapipe
  
You can install the required Python packages using:
```bash
pip install opencv-python numpy tensorflow keras scikit-learn mediapipe
```
## Setup and Installation

- Clone the repository:```git clone https://github.com/SAKU0309/sign-language-recognition.git```

- Navigate to the project directory:```cd sign-language-recognition```

- Install the required dependencies:```pip install -r requirements.txt```

## Usage

1.Ensure your webcam is connected and functioning.
2.Run the script:(It will start the webcam that will collect the images in dataset folder. And can swtich to other gesture by pressing q )
```base
python collectdataset.py
```
3.Run the script:(It will train the model based on your gesture )
```base
python training.py
```
4.Run the main script:(It will open the window that will recognise the gesture of you hand)
```base
python recognize_gesture.py
```
5.Follow the above instructions in the terminal or command prompt for running the application.

## Acknowledgements

- OpenCV for providing the powerful computer vision library.
- TensorFlow and Keras for enabling deep learning and model training.
- scikit-learn for machine learning tools and utilities.
- MediaPipe for advanced hand tracking and gesture recognition capabilities.

## Output
![photo-collage png (1)](https://github.com/user-attachments/assets/1bf7d41c-33a5-4107-9b17-7d255fd9f7fb)



## Contributor
- @SAKU0309
- @Sanjuusanjuu
- @UjjwallS
