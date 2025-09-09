import json
import os

# Image dimensions (width, height)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Directory paths
input_label_dir = r'D:\GuideDog Data\Synthetic Data\Set 3\Labels'  # Folder containing your JSON files
output_label_dir = r'D:\GuideDog Data\Synthetic Data\Set 3\Annotations'  # Folder where YOLO labels will be saved

if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

def convert_bbox_to_yolo_format(top_left, bottom_right, img_width, img_height):
    # Extract coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Calculate center of the box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    
    # Calculate width and height of the box
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    # Normalize coordinates by image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height

def process_json_file(json_file_path, output_file_path):
    # Open the JSON file and load its content
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Open output file for writing YOLO annotations
    with open(output_file_path, 'w') as out_file:
        # Loop over each object in the JSON file
        for obj in data['objects']:
            # Check if object is tactile-paving
            if obj['class']!= 'tactile block': # or obj['visibility']<=0:
                continue
            # Check if bounding_box exists
            if 'bounding_box' in obj:
                top_left = obj['bounding_box']['top_left']
                bottom_right = obj['bounding_box']['bottom_right']
                
                # Convert bounding box to YOLO format
                x_center, y_center, width, height = convert_bbox_to_yolo_format(
                    top_left, bottom_right, IMAGE_WIDTH, IMAGE_HEIGHT)
                
                # Threshold check to identify actual bounding-box
                if (width <0.25 and height <0.25) or (width>5 or height >5):
                    continue

                # Class ID (e.g., assuming tactile paving is class 0)
                class_id = 0  # You can adjust this according to your classes
                
                # Write to the output file
                out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                

# Process each JSON file in the input directory
for json_file in os.listdir(input_label_dir):
    if json_file.endswith('.json'):
        input_json_path = os.path.join(input_label_dir, json_file)
        output_txt_path = os.path.join(output_label_dir, json_file.replace('.json', '.txt'))
        
        # Convert the JSON annotations to YOLO format and save them
        process_json_file(input_json_path, output_txt_path)

print("Conversion completed!")
