import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm

train_images_path = 'dataset/TrainImages'
augmented_images_path = 'dataset/TrainAugmentedImages'

os.makedirs(augmented_images_path, exist_ok=True)
os.makedirs(f"{augmented_images_path}/pos", exist_ok=True)
os.makedirs(f"{augmented_images_path}/neg", exist_ok=True)

augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),                     # Horizontal flip
    iaa.Affine(rotate=(-15, 15)),        # Random rotations
    iaa.Affine(scale=(0.8, 1.2)),        # Random scaling
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Add Gaussian noise
    iaa.Multiply((0.8, 1.2)),            # Random brightness
    iaa.LinearContrast((0.8, 1.2)),      # Contrast adjustments
])

def augment_and_save(images_path, output_dir):
    for filename in tqdm(os.listdir(images_path)):
        if filename.endswith('.pgm'):
            img_path = os.path.join(images_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            augmented_images = augmentations(images=[img for _ in range(5)])

            base_name, ext = os.path.splitext(filename)
            for i, aug_img in enumerate(augmented_images):
                output_path = os.path.join(output_dir, f"{base_name}_aug_{i}{ext}")
                cv2.imwrite(output_path, aug_img)

augment_and_save(f"{train_images_path}/pos", f"{augmented_images_path}/pos")
augment_and_save(f"{train_images_path}/neg", f"{augmented_images_path}/neg")
