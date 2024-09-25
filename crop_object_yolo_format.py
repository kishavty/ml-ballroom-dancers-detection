from cv2 import cv2
import os

def crop_objects_from_images(image_folder, labels_folder, output_folder):
    """
    Crops objects from images based on YOLOv8 bounding box parameters and saves them to a specified folder.

    Args:
        image_folder (str): Path to the folder containing the images.
        labels_folder (str): Path to the folder containing the corresponding label files in YOLOv8 format.
        output_folder (str): Path to the folder where cropped objects will be saved.

    Notes:
        - If an image or its corresponding label file does not exist, the function will skip that image and display a warning message.
        - Bounding box coordinates are converted from normalized YOLOv8 format to pixel values before cropping.
    """

    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(1, 101):
        image_file = os.path.join(image_folder, f'{i:03}.png')
        label_file = os.path.join(labels_folder, f'{i:03}.txt')
        
        if not os.path.exists(image_file):
            print(f"Image {image_file} does not exist.")
            continue
        
        if not os.path.exists(label_file):
            print(f"Label file {label_file} does not exist.")
            continue
        
        image = cv2.imread(image_file)
        h, w, _ = image.shape
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines):
            # parse lines - format: class center_x center_y width height
            parts = line.strip().split()
            class_id, center_x, center_y, width, height = map(float, parts)
            
            # calculate normalized coordinates to real pixels
            cx, cy = int(center_x * w), int(center_y * h)
            bw, bh = int(width * w), int(height * h)
            
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2
            
            cropped = image[y1:y2, x1:x2]

            output_path = os.path.join(output_folder, f'{i:03}_object_{idx}.jpg')
            cv2.imwrite(output_path, cropped)
            print(f"Cropped object saved at {output_path}")

image_folder = 'image-folder-path'
labels_folder = 'labels-folder-path'
output_folder = 'output-folder-path'

crop_objects_from_images(image_folder, labels_folder, output_folder)
