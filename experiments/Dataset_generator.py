import os
import random
import shutil
import yaml

# Define dataset directories
image_dir = r'D:\GuideDog Data\Synthetic Data\Set 3\Images-resized'  # Directory containing all the images
label_dir =  r'D:\GuideDog Data\Synthetic Data\Set 3\Annotations'  # Directory containing YOLO format label files
output_base_dir = r'D:\GuideDog Data\Synthetic Data\Set 3\YOLO_Dataset_3'  # Output directory for training/validation/test sets

# Define split ratios
train_split = 0.8
val_split = 0.1
test_split = 0.1

# Create output directories
train_image_dir = os.path.join(output_base_dir, 'train/images')
val_image_dir = os.path.join(output_base_dir, 'val/images')
test_image_dir = os.path.join(output_base_dir, 'test/images')
train_label_dir = os.path.join(output_base_dir, 'train/labels')
val_label_dir = os.path.join(output_base_dir, 'val/labels')
test_label_dir = os.path.join(output_base_dir, 'test/labels')

for directory in [train_image_dir, val_image_dir, test_image_dir, train_label_dir, val_label_dir, test_label_dir]:
    os.makedirs(directory, exist_ok=True)

# Get a list of all images and labels
images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
labels = [f.replace('.png', '.txt') for f in images]  # Assumes labels match the image names

# Shuffle the dataset
dataset = list(zip(images, labels))
random.shuffle(dataset)

# Calculate split indices
total_images = len(images)
train_count = int(total_images * train_split)
val_count = int(total_images * val_split)
test_count = total_images - train_count - val_count

# Split the dataset
train_set = dataset[:train_count]
val_set = dataset[train_count:train_count + val_count]
test_set = dataset[train_count + val_count:]

# Function to copy images and labels to respective directories
def copy_dataset(data_set, image_dest, label_dest):
    for image, label in data_set:
        shutil.copy(os.path.join(image_dir, image), os.path.join(image_dest, image))
        shutil.copy(os.path.join(label_dir, label), os.path.join(label_dest, label))

# Copy the datasets
copy_dataset(train_set, train_image_dir, train_label_dir)
copy_dataset(val_set, val_image_dir, val_label_dir)
copy_dataset(test_set, test_image_dir, test_label_dir)

# Create the data.yaml file
data_yaml = {
    'train': os.path.abspath(train_image_dir),
    'val': os.path.abspath(val_image_dir),
    'test': os.path.abspath(test_image_dir),
    'nc': 1,  # Number of classes (change if you have multiple classes)
    'names': ['tactile_paving']  # Change this if you have more classes
}

# Save the data.yaml file
with open(os.path.join(output_base_dir, 'data.yaml'), 'w') as yaml_file:
    yaml.dump(data_yaml, yaml_file, default_flow_style=False)

print("Dataset split and data.yaml file created successfully!")
