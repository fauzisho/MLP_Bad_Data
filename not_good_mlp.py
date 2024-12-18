import numpy as np
from tensorflow.keras.optimizers import SGD
from skimage.io import imread
from skimage.transform import resize
import visualkeras
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Constants
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 40
INPUT_SHAPE = IMAGE_WIDTH * IMAGE_HEIGHT  # Flattened image data

# Function to load and preprocess images
def load_images(filepaths):
    images = []
    for filepath in filepaths:
        img = imread(filepath, as_gray=True)  # Read image
        img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))  # Resize image to 100x40
        img = img.flatten()  # Flatten image
        images.append(img)
    return np.array(images)

# Function to display sample images
def display_sample_images(pos_images, neg_images, n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Display positive samples
        img = imread(pos_images[i], as_gray=True)
        plt.subplot(2, n, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title("Positive")
        plt.axis('off')

        # Display negative samples
        img = imread(neg_images[i], as_gray=True)
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(img, cmap='gray')
        plt.title("Negative")
        plt.axis('off')
    plt.show()

# Load dataset
pos_images = glob.glob('dataset/TrainImages/pos/*.pgm')  # Update with actual path
neg_images = glob.glob('dataset/TrainImages/neg/*.pgm')  # Update with actual path
pos_data = load_images(pos_images)
neg_data = load_images(neg_images)

# Display sample images
display_sample_images(pos_images, neg_images)

# Combine data and labels
X = np.concatenate([pos_data, neg_data])
y = np.array([1] * len(pos_data) + [0] * len(neg_data))

# Visualize class distribution
plt.bar(['Positive', 'Negative'], [len(pos_data), len(neg_data)])
plt.title('Class Distribution')
plt.show()

# Shuffle the dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split the dataset
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Count the number of positive and negative samples in the training and testing sets
train_pos_count = sum(y_train == 1)
train_neg_count = sum(y_train == 0)
test_pos_count = sum(y_test == 1)
test_neg_count = sum(y_test == 0)

# Create bar plots to visualize the counts
categories = ['Positive', 'Negative']
train_counts = [train_pos_count, train_neg_count]
test_counts = [test_pos_count, test_neg_count]

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Bar plot for training set
axs[0].bar(categories, train_counts)
axs[0].set_title('Training Set')
axs[0].set_ylabel('Count')

# Bar plot for testing set
axs[1].bar(categories, test_counts)
axs[1].set_title('Validation Set')
axs[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# Define the model
model_momentum = Sequential()
model_momentum.add(Dense(128, input_dim=INPUT_SHAPE, activation='relu'))
model_momentum.add(Dense(64, activation='relu'))
model_momentum.add(Dense(32, activation='relu'))
model_momentum.add(Dense(1, activation='sigmoid'))

# Define the optimizer
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)

# Compile the model
model_momentum.compile(
    loss='binary_crossentropy',
    optimizer=sgd_optimizer,  # Ensure compatibility with TensorFlow
    metrics=['accuracy']
)

# Define the TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Train the model
history = model_momentum.fit(
    X_train, y_train,
    epochs=13,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback]
)

# Evaluate the model
loss_momentum, accuracy_momentum = model_momentum.evaluate(X_test, y_test)
print(f"Test Accuracy with Momentum: {accuracy_momentum * 100:.2f}%")

# Save the model
model_momentum.save('not_good_mlp_car_detection_model_momentum.h5')

# Visualization of training metrics
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Repeat for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
