# Paint Defect Detection Pipeline

This document outlines the full pipeline for detecting paint defects on aircraft surfaces using a combination of state-of-the-art computer vision models including Grounding DINO, SAM, RT-DETR, and Mask2Former. The final solution includes web deployment via ONNX.js and an LLM agent for improving the dataset.

## 1. Auto-Labeling Images with Grounding DINO + SAM

**Objective:**

Automatically generate bounding boxes and pixel-wise masks for unlabeled images using a prompt-based system.

**Tools:**

- Grounding DINO for prompt-based object detection
- Segment Anything Model (SAM) for segmentation masks

**Steps:**

1. Prepare input images under `data/raw/`
2. Define prompts for defect detection, e.g., "paint defect, sag, discoloration, inclusion"
3. Run the labeling script:
   ```bash
   python scripts/label_data.py --img_dir data/raw --out_dir data/labels
   ```

The script saves YOLO-format bounding box labels to `data/labels/`

## 2. Model Training (RT-DETR)

**Objective:**

Train a high-accuracy object detection model using the labeled dataset.

**Tools:**

- RT-DETR (Real-Time DEtection TRansformer)

**Steps:**

1. Configure dataset in `paint_defect.yaml`
2. Train the model:
   ```bash
   python scripts/train_model.py --data_yaml paint_defect.yaml --epochs 100
   ```

Best model is saved to `models/paint_best.pt` and exported to `web/model.onnx`.

## 3. Object Detection (Inference)

**Objective:**

Run inference on images to detect defects, annotate them visually, and export metadata.

**Steps:**

1. Run the inference script:
   ```bash
   python scripts/agent_infer.py --img_dir data/raw --model models/paint_best.pt --out_dir to_review
   ```

This script:
- Loads the trained model
- Runs inference on each image
- Saves annotated `.jpg` images
- Copies low-confidence cases to `to_review/` for review

## 4. LLM Agent for Data Curation

**Objective:**

Use a Large Language Model (LLM) to sort images from the `to_review` folder into `false_positives` and `false_negatives` to prepare for the next round of training.

**Steps:**

1. Configure your OpenAI API key in the script or as an environment variable.
2. Run the LLM agent script:
   ```bash
   python scripts/llm_agent.py --review_dir to_review
   ```

## 5. Web Deployment with ONNX.js

**Objective:**

Allow users to upload images and run detection directly in a browser.

**Steps:**

1. Place `index.html`, `main.js`, and `model.onnx` in the `web/` directory.
2. Open `web/index.html` in any modern browser.

This end-to-end system includes automated labeling, high-accuracy training, object detection, and easy deployment to the web. Future extensions may include mobile deployment, segmentation refinement, or cloud-based APIs for batch processing.
