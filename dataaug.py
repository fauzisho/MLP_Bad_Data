import os
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm
import random

train_images_path = 'dataset/TrainImages'
augmented_images_path = 'dataset/TrainAugmentedImages'

os.makedirs(augmented_images_path, exist_ok=True)
os.makedirs(f"{augmented_images_path}/pos", exist_ok=True)
os.makedirs(f"{augmented_images_path}/neg", exist_ok=True)

def get_random_augmentations():
    return iaa.Sequential([
        iaa.Fliplr(random.uniform(0.3, 0.7)),                    # Random horizontal flip probability
        iaa.Affine(rotate=random.uniform(-10, 20)),              # Random rotation between -10° and 20°
        iaa.Affine(scale=(random.uniform(0.7, 1.3),              # Random scaling (70% to 130%)
                    random.uniform(0.7, 1.3))),
        iaa.AdditiveGaussianNoise(scale=(0, random.uniform(0.02, 0.08) * 255)),  # Random noise
        iaa.Multiply((random.uniform(0.7, 1.3))),                # Random brightness (70% to 130%)
        iaa.LinearContrast((random.uniform(0.7, 1.3)))           # Random contrast adjustments
    ])

def augment_and_save(images_path, output_dir, num_augments=20):
    for filename in tqdm(os.listdir(images_path)):
        if filename.endswith('.pgm'):
            img_path = os.path.join(images_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            for i in range(num_augments):
                augmentations = get_random_augmentations()
                augmented_img = augmentations(image=img)

                base_name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{base_name}_aug_{i}{ext}")
                cv2.imwrite(output_path, augmented_img)

augment_and_save(f"{train_images_path}/pos", f"{augmented_images_path}/pos", num_augments=20)
augment_and_save(f"{train_images_path}/neg", f"{augmented_images_path}/neg", num_augments=20)
