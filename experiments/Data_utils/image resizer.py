import cv2
import os

# Paths
input_dir = r'D:\GuideDog Data\Synthetic Data\seg\images'  # Replace with the folder containing 1920x1080 images
output_dir = r'D:\GuideDog Data\Synthetic Data\seg\images_resized'  # Replace with the folder to save resized images

# Target dimensions
target_width = 1024  # Adjust as needed
target_height = 1024  # Adjust as needed

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Resize images
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Process only image files
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('jpg', 'png'))
        
        # Load the image
        image = cv2.imread(input_path)
        
        # Resize the image
        resized_image = cv2.resize(image, (target_width, target_height))
        
        # Save the resized image
        cv2.imwrite(output_path, resized_image)

print("All images have been resized and saved successfully.")