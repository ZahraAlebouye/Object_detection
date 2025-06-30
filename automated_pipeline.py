import os
import shutil
import argparse

# Import functions from our existing scripts
from agent_infer import infer
from llm_agent import sort_reviewed_images, API_KEY
from label_data import label_images
from train_model import train

def clear_directory(directory):
    """Deletes all files and subdirectories in a given directory."""
    if os.path.exists(directory):
        print(f"Clearing directory: {directory}")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def curate_data(review_dir, data_dir, labels_dir):
    """
    Curates the dataset based on LLM feedback.
    - Moves false negatives to the main data folder to be re-labeled.
    - Creates empty label files for false positives to correct the model.
    """
    print("\n--- Starting Data Curation ---")
    
    false_neg_dir = os.path.join(review_dir, 'false_negatives')
    false_pos_dir = os.path.join(review_dir, 'false_positives')

    # Process false negatives: Images with defects the model missed.
    if os.path.exists(false_neg_dir) and os.listdir(false_neg_dir):
        print(f"Processing {len(os.listdir(false_neg_dir))} false negatives...")
        # Move the images to the main data directory to be included in training.
        for fn in os.listdir(false_neg_dir):
            shutil.copy(os.path.join(false_neg_dir, fn), os.path.join(data_dir, fn))
        
        # Re-run auto-labeling to generate correct labels for these new images.
        print("Running auto-labeling for new false negative images...")
        label_images(img_dir=data_dir, out_dir=labels_dir, prompt='paint defect, sag, discoloration, inclusion', box_threshold=0.35, text_threshold=0.25)

    # Process false positives: Images the model flagged incorrectly.
    if os.path.exists(false_pos_dir) and os.listdir(false_pos_dir):
        print(f"Processing {len(os.listdir(false_pos_dir))} false positives...")
        for fp in os.listdir(false_pos_dir):
            # Copy the image to the main data folder if it's not already there.
            if not os.path.exists(os.path.join(data_dir, fp)):
                shutil.copy(os.path.join(false_pos_dir, fp), os.path.join(data_dir, fp))
            
            # Create an empty label file for it. This teaches the model there's no defect.
            label_name = os.path.splitext(fp)[0] + '.txt'
            with open(os.path.join(labels_dir, label_name), 'w') as f:
                pass # Create an empty file
        print("Created empty labels for false positives.")
    
    print("--- Data Curation Complete ---")

def run_pipeline(initial_model, new_images_dir, data_yaml, epochs):
    """
    Runs the full automated inference, curation, and retraining pipeline.
    """
    # Define paths
    review_dir = 'to_review'
    inference_output_dir = 'inference_output'
    raw_data_dir = 'data/raw'
    labels_dir = 'data/labels'

    # --- Step 1: Run Inference ---
    print("--- Step 1: Running Inference ---")
    clear_directory(review_dir)
    clear_directory(inference_output_dir)
    infer(
        img_dir=new_images_dir,
        model_path=initial_model,
        out_dir=inference_output_dir,
        confidence_threshold=0.5 # This threshold determines what goes to the LLM
    )
    print("--- Inference Complete ---")

    # --- Step 2: LLM Curation ---
    review_image_count = len(os.listdir(review_dir)) if os.path.exists(review_dir) else 0
    if review_image_count > 0:
        print(f"\n--- Step 2: Found {review_image_count} images for review. Running LLM Vision Agent... ---")
        sort_reviewed_images(review_dir=review_dir)
        print("--- LLM Curation Complete ---")

        # --- Step 3: Data Curation & Re-labeling ---
        curate_data(review_dir=review_dir, data_dir=raw_data_dir, labels_dir=labels_dir)
        
        # --- Step 4: Retrain the Model ---
        print("\n--- Step 4: Retraining the Model with Curated Data ---")
        train(data_yaml=data_yaml, epochs=epochs)
        print("--- Automated Pipeline Run Complete ---")
    else:
        print("\nNo images required review. Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Paint Defect Detection Pipeline")
    parser.add_argument('--model', type=str, default='models/paint_defect_run/weights/best.pt', help='Path to the initial model.')
    parser.add_argument('--new_images', type=str, default='data/raw', help='Directory with new images to process for the feedback loop.')
    parser.add_argument('--data_yaml', type=str, default='paint_defect.yaml', help='Path to the data YAML file.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for retraining.')
    
    args = parser.parse_args()

    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set. This is required for the LLM agent.")
    elif not os.path.exists(args.model):
        print(f"Error: Initial model not found at {args.model}.")
        print("Please train an initial model first using 'python scripts/train_model.py'")
    else:
        run_pipeline(args.model, args.new_images, args.data_yaml, args.epochs)
