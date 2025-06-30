import os
import argparse
import torch
from ultralytics import RTDETR

# --- Main Training Logic ---

def train(data_yaml, epochs, model_name='rtdetr-l.pt'):
    """Trains an RT-DETR model and exports it to ONNX."""
    # Load a pretrained RT-DETR model
    model = RTDETR(model_name)

    # Train the model
    print(f"Starting training for {epochs} epochs...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        project='models',
        name='paint_defect_run',
        exist_ok=True
    )

    print("Training complete.")

    # Find the best model weights
    best_model_path = os.path.join('models', 'paint_defect_run', 'weights', 'best.pt')

    # --- Export to ONNX ---
    # Note: The export might require a specific opset version depending on your environment.
    # We are creating a dummy input that matches the expected input shape.
    if os.path.exists(best_model_path):
        print(f"Exporting the best model from {best_model_path} to ONNX...")
        # Load the best model for export
        model_to_export = RTDETR(best_model_path)
        
        # Define the output path for the ONNX model
        onnx_output_path = os.path.join('web', 'model.onnx')
        os.makedirs('web', exist_ok=True)

        # Export the model
        model_to_export.export(format='onnx', imgsz=640, opset=12)
        
        # The export command from ultralytics saves it in the same folder as the weights.
        # We need to move it to our web directory.
        exported_onnx_source = os.path.join('models', 'paint_defect_run', 'weights', 'best.onnx')
        if os.path.exists(exported_onnx_source):
            os.rename(exported_onnx_source, onnx_output_path)
            print(f"Model successfully exported to {onnx_output_path}")
        else:
            print(f"Error: Could not find exported ONNX model at {exported_onnx_source}")

    else:
        print(f"Error: Could not find best model at {best_model_path}")

# --- Script Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RT-DETR model for paint defect detection.")
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to the data YAML file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    
    args = parser.parse_args()

    train(args.data_yaml, args.epochs)
