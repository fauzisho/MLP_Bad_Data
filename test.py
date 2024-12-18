import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt  # Ensure this is included

# Define the paths
model_path = 'car_detection_model_iteration_46.h5'  # Path to the saved model
test_data_path = 'dataset/TrainImages'  # Dataset to test on

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

# Load the test data
print("Loading test data...")
X_test, y_test = load_data(test_data_path)

# Load the pre-trained model
print(f"Loading model from {model_path}...")
model = load_model(model_path)

# Evaluate the model on the test data
print("Evaluating the model...")
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate accuracy
accuracy = np.mean(y_pred.flatten() == y_test)
print(f"Accuracy on the test dataset: {accuracy * 100:.2f}%")

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Car', 'Car'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for {model_path}")
plt.show()
