import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import supervision as sv
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_binary_masks(ground_truth_mask, predicted_mask):

    # Ensure masks are boolean or can be treated as such
    ground_truth_mask = ground_truth_mask.astype(bool)
    predicted_mask = predicted_mask.astype(bool)

    # Calculate TP, FP, TN, FN
    tp = np.sum(np.logical_and(ground_truth_mask, predicted_mask))
    tn = np.sum(np.logical_and(~ground_truth_mask, ~predicted_mask))
    fp = np.sum(np.logical_and(~ground_truth_mask, predicted_mask))
    fn = np.sum(np.logical_and(ground_truth_mask, ~predicted_mask))

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "IoU": iou,
    }

# use bfloat16 for the entire notebook
# from Meta notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

checkpoint = "/content/drive/MyDrive/GuideDog-Synthetic-Data/Weights/SAM2.1/S+R_checkpoint_100.pt" # Your preferred checkpoint
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
mask_generator = SAM2AutomaticMaskGenerator(sam2)


# validation_set = os.listdir("/content/drive/MyDrive/GuideDog-Synthetic-Data/SAM2_Dataset")
input_dir = "/content/drive/MyDrive/GuideDog-Synthetic-Data/Dataset/test-v1"
validation_set = [img for img in os.listdir(input_dir) if img.endswith(".jpg")]
shuffled_validation_set = random.sample(validation_set, len(validation_set))
output_dir = "/content/pred_S+R"  # Save predicted masks

ctr=1
Accuracy = []
Precision = []
Recall = []
F1_Score = []
IoU = []

#image = random.choice([img for img in validation_set if img.endswith(".jpg")])
for image_name in shuffled_validation_set: 
  image_path = os.path.join(input_dir, image_name)
  opened_image = np.array(Image.open(image_path).convert("RGB"))

  mask_path = os.path.join(input_dir, "masks/" +image_name.replace(".jpg", "_mask.jpg"))
  ground_truth_mask = Image.open(mask_path).convert("L")
  result = mask_generator.generate(opened_image)
  #print(result)
  #print(result[0]["segmentation"].shape)

  mask = np.full((360,640),False) #result[0]["segmentation"]
  for m in result:
    mask|=np.array(m["segmentation"])

  binary_data = mask.astype(np.uint8)
  binary_image_array = binary_data * 255
  pred_binary_image= Image.fromarray(binary_image_array)
  pred_binary_image.save(os.path.join(output_dir, image_name.replace(".jpg", "_pred_S+R_mask.jpg")))

  metrics = evaluate_binary_masks(np.array(ground_truth_mask), np.array(pred_binary_image))
  Accuracy.append(metrics["Accuracy"])
  Precision.append(metrics["Precision"])
  Recall.append(metrics["Recall"])
  F1_Score.append(metrics["F1-Score"])
  IoU.append(metrics["IoU"])

  # VIEW COLORED MASKS OVERLAYED ON ORIGINAL IMAGE USING SUPERVISION

  detections = sv.Detections.from_sam(sam_result=result)

  mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
  annotated_image = opened_image.copy()
  annotated_image = mask_annotator.annotate(annotated_image, detections=detections)

  # base_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
  # base_result = mask_generator_base.generate(opened_image)
  # base_detections = sv.Detections.from_sam(sam_result=base_result)
  # base_annotated_image = opened_image.copy()
  # base_annotated_image = base_annotator.annotate(base_annotated_image, detections=base_detections)

  #sv.plot_images_grid(images=[annotated_image, base_annotated_image], titles=["Fine-Tuned SAM-2.1", "Base SAM-2.1"], grid_size=(1, 2))
  sv.plot_images_grid(images=[annotated_image, pred_binary_image, ground_truth_mask], titles=["Fine-Tuned SAM-2.1", "Prediction", "Ground-truth"], grid_size=(1, 3)) # View predictions side-by-side

  images=[opened_image, pred_binary_image, pred_binary_image2, ground_truth_mask]
  titles=["Original", "Prediction_S+R", "Prediction_R", "Ground-truth"] 
  grid_size=(1, 4)
  size = (12,6)
  cmap = "gray"
  
  nrows, ncols = grid_size

  for idx, img in enumerate(images):
      if isinstance(img, Image.Image):
          images[idx] = pillow_to_cv2(img)

  if len(images) > nrows * ncols:
      raise ValueError(
          "The number of images exceeds the grid size. Please increase the grid size"
          " or reduce the number of images."
      )

  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

  for idx, ax in enumerate(axes.flat):
      if idx < len(images):
          if images[idx].ndim == 2:
              ax.imshow(images[idx], cmap=cmap)
          else:
              ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

          if titles is not None and idx < len(titles):
              ax.set_title(titles[idx])

  
  # Extract filename without extension and construct output path
  
  #output_path = os.path.join(output_dir, f"comparison_{os.path.splitext(image_name)[0]}.jpg") 
  #comparison = np.concatenate((annotated_image, base_annotated_image), axis=1)
  #Image.fromarray(comparison).save(output_path)
  #plt.savefig(output_path, bbox_inches='tight')
  plt.close()  # Close to avoid accumulating figures
  #ctr+=1
  #print(ctr)
  # if ctr == 100:
  #   break

print(f"Average Accuracy: {np.mean(Accuracy)}")
print(f"Average Precision: {np.mean(Precision)}")
print(f"Average Recall: {np.mean(Recall)}")
print(f"Average F1-Score: {np.mean(F1_Score)}")
print(f"Average IoU: {np.mean(IoU)}")


