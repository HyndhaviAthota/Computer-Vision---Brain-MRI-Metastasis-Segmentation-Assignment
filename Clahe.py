import cv2
import os

def apply_clahe_to_image(input_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(input_image)

def preprocess_image_directory(directory_path):
    images_list = []
    for filename in os.listdir(directory_path):
        if not filename.endswith("_mask.tif"):  # Ignore mask files
            img = cv2.imread(os.path.join(directory_path, filename), cv2.IMREAD_GRAYSCALE)
            img = apply_clahe_to_image(img)
            images_list.append(img)
    return images_list
