import cv2
import os

# Paths to input and output directories
input_dir = r'D:\GuideDog Data\Real Data\Crosswalk Eval'  # Replace with your input folder path
output_dir = r'D:\GuideDog Data\Real Data\Crosswalk-Eval-Sliced'  # Replace with your output folder path

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Process only image files
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Get image dimensions
        height, width, _ = image.shape

        # Split image into four quadrants
        top = image[0:1080, 0:width]
        bottom_left = image[1080:height, 0:464]
        bottom_mid = image[1080:height, 728:1192]
        bottom_right = image[1080:height, 1456:width]

        # Save each quadrant as a separate image with a modified filename
        base_filename = os.path.splitext(filename)[0]  # Remove file extension
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_top.jpg"), top)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_bottom_left.jpg"), bottom_left)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_bottom_mid.jpg"), bottom_mid)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_bottom_right.jpg"), bottom_right)

print("All images in the folder have been split and saved successfully.")
