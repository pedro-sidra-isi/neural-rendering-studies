#%%
import cv2
import os
import shutil

def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(variance)
    return variance < threshold

def find_and_move_blurry_images(source_folder, destination_folder, threshold=700):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_path = os.path.join(source_folder, filename)
            if is_blurry(image_path, threshold):
                destination_path = os.path.join(destination_folder, filename)
                shutil.move(image_path, destination_path)
                print(f"Moved blurry image: {filename}")

# Usage
source_folder = '/mnt/d/data/maq_0207_sam/angular/images'
destination_folder = '/mnt/d/data/maq_0207_sam/angular/images_blurry'
threshold = 100  # Adjust the threshold as needed

find_and_move_blurry_images(source_folder, destination_folder, threshold)

# %%

import cv2
import os
import shutil

def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"{image_path} - Variance: {variance}")
    return variance < threshold

def find_and_move_blurry_images(source_folder, destination_folder, threshold=100):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_path = os.path.join(source_folder, filename)
            if is_blurry(image_path, threshold):
                destination_path = os.path.join(destination_folder, filename)
                shutil.move(image_path, destination_path)
                print(f"Moved blurry image: {filename}")

# Usage
source_folder = '/mnt/d/data/maq_0207_sam/angular/images'
destination_folder = '/mnt/d/data/maq_0207_sam/angular/images_blurry'
threshold = 1500  # Adjust the threshold as needed

find_and_move_blurry_images(source_folder, destination_folder, threshold)

# %%
