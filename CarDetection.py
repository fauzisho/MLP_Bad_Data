import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

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
        """Predict whether the image contains a car."""
        preprocessed_img = self.preprocess_image(image_path)
        prediction = self.model.predict(preprocessed_img, verbose=0)[0][0]
        return prediction > threshold  # Return True if car detected, else False

def test_folder_and_analyze(folder_path, detector):
    """
    Test all images in a folder and analyze results.
    Args:
        folder_path (str): Path to test images.
        detector (CarDetector): Instance of CarDetector.
    """
    total_images = 0
    car_detected = 0
    not_detected = []

    print("\nTesting images...")

    # Predict for all images
    for filename in os.listdir(folder_path):
        if filename.endswith('.pgm'):
            total_images += 1
            image_path = os.path.join(folder_path, filename)
            has_car = detector.predict(image_path)

            if has_car:
                car_detected += 1
            else:
                not_detected.append(filename)

    # Display not detected images
    print("\nImages where a car was not detected:")
    for filename in not_detected:
        print(f"Not Detected: {filename}")

    # Generate confusion matrix
    y_true = [True] * total_images  # Ground truth: All images contain cars
    y_pred = [True if f not in not_detected else False for f in os.listdir(folder_path) if f.endswith('.pgm')]

    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Car Detected", "Not Detected"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Summary
    print(f"\nTotal Images: {total_images}")
    print(f"Detected Cars: {car_detected}")
    print(f"Not Detected: {len(not_detected)}")

# Example usage
if __name__ == "__main__":
    # Initialize the detector with your trained model
    model_path = 'car_detection_model_60_20_20.h5'  # Path to trained model
    no_aug_model_path = 'no_aug_car_detection_model_60_20_20.h5'  # Path to trained model
    test_folder = 'dataset/TestImages'  # Folder containing test images

    detector = CarDetector(model_path=model_path)
    test_folder_and_analyze(test_folder, detector)

    print("no augmentation image")
    detector = CarDetector(model_path=no_aug_model_path)
    test_folder_and_analyze(test_folder, detector)