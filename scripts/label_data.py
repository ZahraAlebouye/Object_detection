import os
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Grounding DINO
from groundingdino.util.inference import Model

# segment anything
from segment_anything import build_sam, SamPredictor

import supervision as sv

# --- Constants ---

# GroundingDINO
GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"

# SAM
SAM_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"

# --- Model Loading ---

def load_models():
    """Loads GroundingDINO and SAM models."""
    # Note: You may need to download the model weights first.
    # mkdir weights
    # wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights
    # wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GroundingDINO
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # SAM
    sam = build_sam(checkpoint=SAM_CHECKPOINT_PATH).to(device)
    sam_predictor = SamPredictor(sam)

    return grounding_dino_model, sam_predictor, device

# --- Main Labeling Logic ---

def label_images(img_dir, out_dir, prompt, box_threshold, text_threshold):
    """Processes images in a directory, detects objects, and saves YOLO labels."""
    grounding_dino_model, sam_predictor, device = load_models()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in tqdm(image_paths, desc="Labeling Images"):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect with GroundingDINO
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=[prompt],
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # Filter detections (assuming class_id 0 for all defects)
        # This step is simplified; in a real scenario, you might have multiple classes.
        if len(detections.xyxy) == 0:
            continue

        # Convert to YOLO format
        h, w, _ = image.shape
        yolo_labels = []
        for box in detections.xyxy:
            x_center = (box[0] + box[2]) / 2 / w
            y_center = (box[1] + box[3]) / 2 / h
            width = (box[2] - box[0]) / w
            height = (box[3] - box[1]) / h
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save YOLO label file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(out_dir, f"{base_name}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

    print(f"\nLabeling complete. Labels saved to {out_dir}")

# --- Script Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using GroundingDINO and SAM.")
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with raw images.')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save YOLO labels.')
    parser.add_argument('--prompt', type=str, default='paint defect, sag, discoloration, inclusion', help='Prompt for defect detection.')
    parser.add_argument('--box_threshold', type=float, default=0.35, help='Box confidence threshold.')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='Text similarity threshold.')

    args = parser.parse_args()

    label_images(args.img_dir, args.out_dir, args.prompt, args.box_threshold, args.text_threshold)
