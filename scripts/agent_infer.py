import os
import argparse
import cv2
import shutil
from ultralytics import RTDETR
from supervision import BoxAnnotator, Detections
import numpy as np

# --- Inference Logic ---

def infer(img_dir, model_path, out_dir, confidence_threshold=0.5):
    """Runs inference on images, saves annotated results, and flags low-confidence ones."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load the trained model
    model = RTDETR(model_path)

    # Setup annotator
    box_annotator = BoxAnnotator()

    image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Run inference
        results = model(image)[0]
        detections = Detections.from_ultralytics(results)

        # Separate high and low confidence detections
        low_conf_detections = detections[detections.confidence < confidence_threshold]
        high_conf_detections = detections[detections.confidence >= confidence_threshold]

        # Annotate image with high-confidence detections
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=high_conf_detections)

        # Save the annotated image
        base_name = os.path.basename(image_path)
        output_path = os.path.join(out_dir, base_name)
        cv2.imwrite(output_path, annotated_image)

        # If there are low-confidence detections, copy the original image for review
        if len(low_conf_detections) > 0:
            review_dir = 'to_review'
            if not os.path.exists(review_dir):
                os.makedirs(review_dir)
            shutil.copy(image_path, os.path.join(review_dir, base_name))
            print(f"Copied {base_name} to {review_dir} for review due to low confidence.")

    print(f"Inference complete. Annotated images saved to {out_dir}")

# --- Script Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained RT-DETR model.")
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with images for inference.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model (.pt file).')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save annotated images.')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold to flag images for review.')

    args = parser.parse_args()

    infer(args.img_dir, args.model, args.out_dir, args.confidence_threshold)
