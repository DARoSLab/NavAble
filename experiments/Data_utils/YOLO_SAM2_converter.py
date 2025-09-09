import json
import os

og_img_width= 1920
og_img_height = 1080
input_yolo_label_dir = r"D:\GuideDog Data\Synthetic Data\Set 5\YOLO_Dataset_combined\train\labels" # r"D:\GuideDog Data\Synthetic Data\Set 5\Annotations\label"
output_SAM2_dir = r"D:\GuideDog Data\Synthetic Data\Set 5\SAM2.1_Dataset_combined"

def convert_yolo_to_sam2(yolo_json_path, json_file, n):
    """
    Converts YOLO12 object detection annotations into SAM2 format, preserving object color.
    """
    isEmpty = True
    with open(yolo_json_path, 'r') as f:
        # yolo_data = json.load(f)  # Load YOLO12 annotations

        sam2_annotation = {}  # Store converted annotations

        for i, obj in enumerate(f):
            '''
            # Extract bounding box
            x_min, y_min = ((obj["bounding_box"]["x_min"])/og_img_width)*1024, ((obj["bounding_box"]["y_min"])/og_img_height)*1024
            x_max, y_max = ((obj["bounding_box"]["x_max"])/og_img_width)*1024, ((obj["bounding_box"]["y_max"])/og_img_height)*1024

            width, height = x_max - x_min, y_max - y_min
            area = width * height

            # Store object color (from `object_id`)
            object_color = obj["object_id"]  # RGB color value
            '''

            parts = obj.strip().split()
            if len(parts) == 5:
                isEmpty = False
                #continue
            cls_id = int(parts[0])
            x_c_rel, y_c_rel, w_rel, h_rel = map(float, parts[1:])

            x_c = x_c_rel * 1024
            y_c = y_c_rel * 1024
            width = w_rel * 1024
            height = h_rel * 1024
            x_min = x_c - width / 2
            y_min = y_c - height / 2
            area = width * height

            # Create SAM2 format annotation
            sam2_annotation = {
                "image": {
                    "image_id": n + 1,  # Unique ID
                    "license": 1,
                    "file_name": json_file.replace('.txt', '.jpg'), # obj.get("image_file", f"image_{i+1}.jpg"),  # Assuming filename exists
                    "height": 1024,  # Modify based on actual image size
                    "width": 1024,
                    "date_captured": "2025-05-24T11:35:58+00:00"
                },
                "annotations": [
                    {
                        "id": i + 1,
                        "bbox": [x_min, y_min, width, height],  # Convert to [x, y, width, height]
                        "area": area,
                        "segmentation": {
                            "counts": "",  # Empty for now, can be generated later
                            "size": [1024, 1024]
                        },
                        "object_color": [153, 108, 6]  # object_color  # Store the object's RGB color
                    }
                ]
            }

    if not isEmpty:
        #sam2_annotations.append(sam2_annotation)
        output_json_path = os.path.join(output_SAM2_dir, json_file.replace('.txt', '.json')) #json_file.replace('.json', '_rgb.json'))
    # Save the converted annotations
        with open(output_json_path, 'w') as f:
            json.dump(sam2_annotation, f)

    # print(f"Converted {len(yolo_data)} YOLO12 annotations to SAM2 format with color information.")

# Example usage:
# convert_yolo_to_sam2("yolo12_annotations.json", "sam2_annotations.json")
# Process each JSON file in the input directory
n=0
for json_file in os.listdir(input_yolo_label_dir):
    if json_file.endswith('.txt'):
        if json_file.
        input_json_path = os.path.join(input_yolo_label_dir, json_file)
        
        
        # Convert the JSON annotations to YOLO format and save them
        # process_json_file(input_json_path, output_txt_path)
        convert_yolo_to_sam2(input_json_path, json_file, n)
        n+=1

print("Conversion completed!")