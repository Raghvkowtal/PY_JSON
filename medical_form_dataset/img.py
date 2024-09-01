import os
import cv2
import json
import easyocr
from datetime import datetime

# Initialize EasyOCR reader with GPU support
import torch
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU if available

# Paths
dataset_path = 'D:\\ftest - Copy\\medical_form_dataset'
images_path = os.path.join(dataset_path, 'images')
annotations_path = os.path.join(dataset_path, 'annotations')
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Initialize an empty dictionary to hold all extracted texts
all_extracted_texts = {}
accuracies = []

# Preprocessing function
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Helper function to calculate accuracy based on letter identification
def calculate_accuracy(true_text, detected_text):
    true_text = true_text.lower().strip()
    detected_text = detected_text.lower().strip()
    
    if not true_text:
        return 0.0 if detected_text else 100.0

    # Calculate the number of correctly identified characters
    correct_chars = sum(1 for a, b in zip(true_text, detected_text) if a == b)
    total_chars = len(true_text)
    
    # Add remaining characters from true text if detected_text is shorter
    if len(detected_text) < len(true_text):
        total_chars += len(true_text) - len(detected_text)

    if total_chars == 0:
        return 0.0

    return (correct_chars / total_chars) * 100

# Process each image and its corresponding annotation
for annotation_file in os.listdir(annotations_path):
    if annotation_file.endswith('.json'):
        # Corresponding image file
        image_file = annotation_file.replace('.json', '.jpg')
        image_path = os.path.join(images_path, image_file)
        annotation_path = os.path.join(annotations_path, annotation_file)
        
        # Debugging: Print paths
        print(f"Processing image: {image_path}")
        print(f"Processing annotation: {annotation_path}")

        # Check if image file exists
        if not os.path.isfile(image_path):
            print(f"Error: The image file at {image_path} does not exist.")
            continue

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: The image file at {image_path} could not be opened.")
            continue
        
        # Load annotations
        with open(annotation_path) as f:
            annotation_data = json.load(f)
        
        # Process each ROI
        for annotation in annotation_data['annotations']:
            x1, y1 = annotation['coordinates']['x1'], annotation['coordinates']['y1']
            x2, y2 = annotation['coordinates']['x2'], annotation['coordinates']['y2']
            label = annotation['label']
            true_text = annotation.get('true_text', '')  # Ensure you have 'true_text' in annotations

            # Ensure ROI coordinates are within image bounds
            x1, x2 = max(0, x1), min(img.shape[1], x2)
            y1, y2 = max(0, y1), min(img.shape[0], y2)

            # Extract ROI
            roi = img[y1:y2, x1:x2]

            # Save the ROI as an image file
            roi_filename = os.path.join(output_folder, f'{label}_{image_file}')
            cv2.imwrite(roi_filename, roi)
            print(f"Saved ROI: {roi_filename}")

            # Perform OCR
            results = reader.readtext(roi, detail=1)
            if results:
                roi_text = ' '.join([result[1] for result in results])
                extracted_texts = roi_text.strip()
                print(f"Detected text in ROI '{label}': {extracted_texts}")
            else:
                extracted_texts = 'No text detected'
                print(f"No text detected in ROI '{label}'")

            # Calculate accuracy
            accuracy = calculate_accuracy(true_text, extracted_texts)
            accuracies.append(accuracy)

        # Save the extracted texts to JSON
        all_extracted_texts[image_file] = extracted_texts

# Save all extracted texts to a single JSON file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_filename = f'all_extracted_texts_{timestamp}.json'
json_output_path = os.path.join(output_folder, json_filename)

with open(json_output_path, 'w') as json_file:
    json.dump(all_extracted_texts, json_file, indent=4)

print(f"All extracted texts are saved to {json_output_path}")

# Calculate and display total accuracy
if accuracies:
    total_accuracy = sum(accuracies) / len(accuracies)
else:
    total_accuracy = 0.0

print(f"Total accuracy: {total_accuracy:.2f}%")
