# GENERATE GROUND TRUTH BINARY MASKS

from PIL import Image
import numpy as np
import os, json
from pycocotools import mask

input_dir= r"D:\GuideDog Data\Real Data\Real Tactile Data\test-v3"
output_dir= r"D:\GuideDog Data\Real Data\Real Tactile Data\test-v3\masks"

for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(input_dir, filename), 'r') as f:
        data = json.load(f)
   
    # if len(data["annotations"])>1:
    #     print(filename)
    
    seg = data["annotations"][0]["segmentation"]
    m = mask.decode(seg)

    for ann in data["annotations"]:
        seg= ann["segmentation"]
        m|=mask.decode(seg)

    Image.fromarray(m*255).save(os.path.join(output_dir, filename.replace(".json",".png")))