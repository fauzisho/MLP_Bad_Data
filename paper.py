import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Paths to positive and negative samples
pos_path = os.path.join("dataset/TrainAugmentedImages/", 'pos')
neg_path = os.path.join("dataset/TrainAugmentedImages/", 'neg')

# Parameters
img_height, img_width = 84, 28  # Resize dimensions from the paper

# Function to load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.endswith('.pgm'):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load positive and negative samples
pos_images, pos_labels = load_images_from_folder(pos_path, 1)
neg_images, neg_labels = load_images_from_folder(neg_path, 0)

# Combine and shuffle the data
X = np.concatenate((pos_images, neg_images), axis=0)
y = np.concatenate((pos_labels, neg_labels), axis=0)

# Normalize the images
X = X / 255.0  # Scale pixel values to [0, 1]
X = np.expand_dims(X, axis=-1)  # Add channel dimension

# One-hot encode labels
y = to_categorical(y, num_classes=2)

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

# Define the model based on the paper
model = Sequential([
    Conv2D(64, (9, 9), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D((2, 2)),  # Adjust pooling size to ensure valid dimensions
    Conv2D(8, (3, 7), activation='relu'),  # Reduced filter size to avoid negative output dimensions
    Flatten(),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
train_generator = data_gen.flow(X, y, batch_size=32)
model.fit(train_generator, epochs=25, steps_per_epoch=len(X) // 32)

# Save the model
model.save('car_detection_model.h5')

# Test set evaluation
def preprocess_test_images(test_folder):
    test_images = []
    for filename in os.listdir(test_folder):
        filepath = os.path.join(test_folder, filename)
        if filename.endswith('.pgm'):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))
            test_images.append(img)
    test_images = np.array(test_images) / 255.0  # Normalize
    test_images = np.expand_dims(test_images, axis=-1)  # Add channel dimension
    return test_images

# Load the trained model
model = load_model('car_detection_model.h5')

# Path to the test images
test_images_path = 'dataset/TestImages'

# Preprocess the test set
test_X = preprocess_test_images(test_images_path)

# Make predictions
test_predictions = model.predict(test_X)

# Print predictions for inspection
print("Predictions:", test_predictions)
