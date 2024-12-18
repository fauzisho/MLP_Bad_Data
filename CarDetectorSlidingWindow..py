import cv2
import numpy as np
from tensorflow.keras.models import load_model

class CarDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)  # Load trained MLP model
        self.input_width = 80  # Width of input image
        self.input_height = 50  # Height of input image

    def sliding_window(self, image, step_size, window_size):
        """Generate patches using sliding window."""
        for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
            for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def preprocess_patch(self, patch):
        """Flatten and normalize a patch for MLP input."""
        patch_resized = cv2.resize(patch, (self.input_width, self.input_height))  # Resize to 80x50
        patch_flattened = patch_resized.flatten() / 255.0  # Normalize to [0, 1]
        return np.expand_dims(patch_flattened, axis=0)  # Add batch dimension

    def predict(self, image_path, step_size=16):
        """Run sliding window detection."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        window_size = (self.input_width, self.input_height)
        bounding_boxes = []

        # Slide window across the image
        for (x, y, patch) in self.sliding_window(image, step_size, window_size):
            processed_patch = self.preprocess_patch(patch)  # Preprocess the patch
            prediction = self.model.predict(processed_patch, verbose=0)[0][0]

            if prediction > 0.5:  # Car detected
                print(f"Car detected at: x={x}, y={y}, w={window_size[0]}, h={window_size[1]}")
                bounding_boxes.append((x, y, x + window_size[0], y + window_size[1]))  # Save box coordinates

        print("Total Bounding Boxes Detected:", len(bounding_boxes))
        return image, bounding_boxes

    def draw_bounding_boxes(self, image, bounding_boxes, step_size=16):
        """Draw black sliding windows and green bounding boxes for detections."""
        # Convert grayscale to BGR to enable colored borders
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()

        # Define window size
        window_size = (self.input_width, self.input_height)

        # Draw all sliding windows with black borders first
        for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
            for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
                cv2.rectangle(image_bgr, (x, y), (x + window_size[0], y + window_size[1]), (0, 0, 0), 1)

        # Draw green borders only for detected bounding boxes
        for (x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green border

        return image_bgr

# Example usage
detector = CarDetector(model_path='car_detection_model_60_20_20.h5')
# detector = CarDetector(model_path='car_detection_cnn_lenet5.h5')
image_path = 'dataset/TestImages/test-19.pgm'
# image_path = 'dataset/TrainImages/neg/neg-60.pgm'

# Predict and detect bounding boxes
image, bounding_boxes = detector.predict(image_path)
if bounding_boxes:
    print(f"Detected {len(bounding_boxes)} car(s).")
    image_with_boxes = detector.draw_bounding_boxes(image, bounding_boxes)
    cv2.imshow("Car Detection with Bounding Boxes", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Detected {len(bounding_boxes)} car(s).")
    image_with_boxes = detector.draw_bounding_boxes(image, bounding_boxes)
    cv2.imshow("Car Detection with Bounding Boxes", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("No cars detected.")
