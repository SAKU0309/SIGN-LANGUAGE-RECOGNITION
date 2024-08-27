import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Parameters
image_size = 64

# Correct path to the dataset
data_dir = r"C:\Users\My PC\Desktop\sign language\dataset"  # Use raw string literal or double backslashes

# Preprocess the dataset
def load_data(data_dir):
    images = []
    labels = []
    for label, gesture in enumerate(os.listdir(data_dir)):
        gesture_dir = os.path.join(data_dir, gesture)
        for image_name in (os.listdir(gesture_dir)):
            image_path = os.path.join(gesture_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (image_size, image_size))
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

images, labels = load_data(data_dir)
images = images.reshape(images.shape[0], image_size, image_size, 1)
images = images / 255.0
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(os.listdir(data_dir)), activation='softmax')  # Number of gestures
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('gesture_model.h5')