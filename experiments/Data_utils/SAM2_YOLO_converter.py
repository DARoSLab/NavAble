import os
import json

# Paths
sam2_dir = r"D:\GuideDog Data\Synthetic Data\Set 4\SAM2_Dataset"
yolo_output_dir = r"D:\GuideDog Data\Synthetic Data\Set 4\yolo_labels"
os.makedirs(yolo_output_dir, exist_ok=True)

# Fixed class ID if there's only one class (e.g., "crosswalk" or "tactile paving")
CLASS_ID = 0

# Process each SAM2 JSON
for filename in os.listdir(sam2_dir):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(sam2_dir, filename), 'r') as f:
        data = json.load(f)

    yolo_lines = []
    if len(data)!=0:
        
        image_info = data["image"]
        annotations = data["annotations"]
        img_width = image_info["width"]
        img_height = image_info["height"]



        for ann in annotations:
            x_min, y_min, width, height = ann["bbox"]
            x_center = x_min + width / 2
            y_center = y_min + height / 2

            # Normalize
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            # Format line
            yolo_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)

    # Write YOLO label file
    # base_name = os.path.splitext(image_info["file_name"])[0]
    # label_path = os.path.join(yolo_output_dir, f"{base_name}.txt")
    label_name = filename.replace('json', 'txt')
    label_path = os.path.join(yolo_output_dir, label_name)
    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_lines))

print("✅ Conversion to YOLO format complete.")
