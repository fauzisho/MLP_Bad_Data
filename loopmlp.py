import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

augmented_images_path = 'dataset/TrainAugmentedImages'

class CarDetector:
    def __init__(self, model_path, input_width=100, input_height=40):
        """Initialize the CarDetector with the model and input size."""
        self.model = load_model(model_path)
        self.input_width = input_width
        self.input_height = input_height

    def preprocess_image(self, image_path):
        """Preprocess an image: resize, normalize, and flatten."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        img_normalized = img_resized / 255.0
        img_flattened = img_normalized.flatten()
        return np.expand_dims(img_flattened, axis=0)  # Add batch dimension

    def predict(self, image_path, threshold=0.5):
        """
        Predict whether the image contains a car and return the confidence score.
        Args:
            image_path (str): Path to the input image.
            threshold (float): Confidence threshold to decide if the image contains a car.
        Returns:
            tuple: (confidence_score, is_car_detected)
        """
        preprocessed_img = self.preprocess_image(image_path)
        prediction_score = self.model.predict(preprocessed_img, verbose=0)[0][0]
        return prediction_score, prediction_score > threshold

def load_data(base_path):
    """Load image data from positive and negative folders."""
    X, y = [], []
    for label, folder in enumerate(['pos', 'neg']):
        folder_path = os.path.join(base_path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.pgm'):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_flattened = img.flatten() / 255.0  # Flatten and normalize
                X.append(img_flattened)
                y.append(label)
    return np.array(X), np.array(y)

def train_and_retrain_loop(train_folder, test_folder, input_dim, target_accuracy=0.99, max_retrain_iterations=100):
    """
    Train and retrain the model until the detection accuracy on the test dataset is >= target_accuracy.
    Args:
        train_folder (str): Path to the training dataset folder.
        test_folder (str): Path to the test dataset folder.
        input_dim (int): Input dimension of the flattened images.
        target_accuracy (float): Target detection accuracy (default is 0.99).
        max_retrain_iterations (int): Maximum number of retraining iterations.
    """
    retrain_iteration = 0
    current_accuracy = 0

    while current_accuracy < target_accuracy and retrain_iteration < max_retrain_iterations:
        print(f"\n--- Retraining Iteration: {retrain_iteration + 1} ---")

        # Load and split the training data
        X, y = load_data(train_folder)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # Define the MLP model
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary output
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        print("Training the model...")
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

        # Save the trained model with incremented name
        model_path = f'car_detection_model_iteration_{retrain_iteration + 1}.h5'
        model.save(model_path)
        print(f"Model saved as '{model_path}'")

        # Test the model on the test folder
        detector = CarDetector(model_path=model_path)
        total_images, car_detected = test_folder_and_analyze(test_folder, detector)

        # Calculate accuracy
        current_accuracy = car_detected / total_images
        print(f"\nCurrent Accuracy on test folder: {current_accuracy * 100:.2f}%")

        retrain_iteration += 1

    if current_accuracy >= target_accuracy:
        print("\nTarget accuracy achieved!")
    else:
        print("\nMaximum retraining iterations reached. Consider increasing the training data or modifying the model.")

def test_folder_and_analyze(folder_path, detector):
    """
    Test all images in a folder and analyze results.
    Args:
        folder_path (str): Path to test images.
        detector (CarDetector): Instance of CarDetector.
    Returns:
        tuple: (total_images, car_detected)
    """
    total_images = 0
    car_detected = 0

    print("\nTesting images...")

    for filename in os.listdir(folder_path):
        if filename.endswith('.pgm'):
            total_images += 1
            image_path = os.path.join(folder_path, filename)
            _, has_car = detector.predict(image_path)
            if has_car:
                car_detected += 1

    print(f"Total Images: {total_images}")
    print(f"Detected Cars: {car_detected}")

    return total_images, car_detected

if __name__ == "__main__":
    train_folder = 'dataset/TrainAugmentedImages'  # Training dataset folder
    test_folder = 'dataset/TestImages'  # Test dataset folder

    # Load data to get input dimension
    X, y = load_data(train_folder)
    input_dim = X.shape[1]  # Flattened image size

    # Start training and retraining loop
    train_and_retrain_loop(train_folder, test_folder, input_dim)
