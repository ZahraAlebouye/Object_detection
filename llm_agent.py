import os
import argparse
import base64
import requests
import shutil
from tqdm import tqdm

# --- LLM Configuration ---
# It's best to set this as an environment variable
API_KEY = os.environ.get("OPENAI_API_KEY")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# --- Image Encoding ---

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- LLM Analysis ---

def analyze_image(image_path):
    """Sends an image to the LLM for analysis and returns the classification."""
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")

    base64_image = encode_image(image_path)

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze the attached image for paint defects on a surface. Is there a clear paint defect such as a sag, discoloration, or inclusion visible? Respond with only 'true' if a defect is present, and 'false' if no defect is present. Do not add any other text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    content = response.json()['choices'][0]['message']['content'].lower().strip()
    
    if 'true' in content:
        return 'false_negative' # The model missed this, so it's a false negative for the *next* run
    elif 'false' in content:
        return 'false_positive' # The model flagged this, but it's not a defect
    else:
        return 'uncertain'

# --- Main Logic ---

def sort_reviewed_images(review_dir):
    """Sorts images from the review directory into false positives and negatives."""
    false_pos_dir = os.path.join(review_dir, 'false_positives')
    false_neg_dir = os.path.join(review_dir, 'false_negatives')
    uncertain_dir = os.path.join(review_dir, 'uncertain')

    os.makedirs(false_pos_dir, exist_ok=True)
    os.makedirs(false_neg_dir, exist_ok=True)
    os.makedirs(uncertain_dir, exist_ok=True)

    image_paths = [os.path.join(review_dir, f) for f in os.listdir(review_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in tqdm(image_paths, desc="Analyzing Images with LLM"):
        try:
            classification = analyze_image(image_path)
            base_name = os.path.basename(image_path)

            if classification == 'false_positive':
                shutil.move(image_path, os.path.join(false_pos_dir, base_name))
            elif classification == 'false_negative':
                shutil.move(image_path, os.path.join(false_neg_dir, base_name))
            else:
                shutil.move(image_path, os.path.join(uncertain_dir, base_name))
                
        except Exception as e:
            print(f"\nCould not process {os.path.basename(image_path)}: {e}")

    print(f"\nLLM analysis complete. Images sorted in {review_dir}.")

# --- Script Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use an LLM to sort images for retraining.")
    parser.add_argument('--review_dir', type=str, required=True, help='Directory with images for review.')
    
    args = parser.parse_args()

    sort_reviewed_images(args.review_dir)
