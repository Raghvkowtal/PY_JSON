import cv2 
import easyocr
import os
import numpy as np
import json
import torch
from datetime import datetime

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

# Initialize EasyOCR reader with GPU support
reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU if available

# Define the file path
file_path = 'D:\\ftest - Copy\\image .jpg'

# Load the image
img = cv2.imread(file_path)

# Check if image was loaded successfully
if img is None:
    print(f"Error: The image file at {file_path} could not be opened. Please check the file path.")
    exit(1)

# Resize the image to a higher resolution (if needed, adjust based on your image size)
img = cv2.resize(img, (2000, 2000))  # Increase resolution

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Increase contrast using histogram equalization
img_hist_eq = cv2.equalizeHist(img_gray)

# Apply GaussianBlur to reduce noise
img_blur = cv2.GaussianBlur(img_hist_eq, (5, 5), 0)

# Apply adaptive thresholding
img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

# Perform morphological operations (optional, for noise reduction)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Rectangular kernel
img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

# Define ROIs with coordinates and descriptive names
rois = [
[(257, 233), (598, 275), 'patient_full_name'],
    [(724, 236), (859, 277), 'date_of_birth'],
    [(552, 278), (983, 313), 'email'],
    [(211, 315), (823, 346), 'home_address'],
    [(863, 314), (983, 346), 'apt'],
    [(155, 348), (520, 381), 'city'],
    [(555, 349), (607, 383), 'state'],
    [(662, 350), (761, 385), 'zipcode'],
    [(802, 351), (968, 385), 'phone1'],
    [(301, 384), (760, 418), 'emergency_contact'],
    [(802, 386), (968, 417), 'phone2'],
    [(225, 418), (381, 454), 'primary_insurance'],
    [(431, 419), (538, 456), 'policy'],
    [(656, 420), (817, 455), 'secondary_insurance'],
    [(656, 420), (817, 455), 'policy_number'],
    [(864, 422), (984, 453), 'secondary_phone']
]

# Create a folder to save the ROIs if it doesn't exist
output_folder = 'Extracted_Json'
os.makedirs(output_folder, exist_ok=True)

# Initialize an empty dictionary to hold extracted texts with default empty values
extracted_texts = {
    "patient_full_name": "",
    "date_of_birth": "",
    "email": "",
    "home_address": "",
    "apt": "",
    "city": "",
    "state": "",
    "zipcode": "",
    "phone1": "",
    "emergency_contact": "",
    "phone2": "",
    "primary_insurance": "",
    "policy": "",
    "secondary_insurance": "",
    "policy_number": "",
    "secondary_phone": ""
}

# Variables to track total accuracy
total_confidences = []
total_texts = []

# Loop through each ROI
for i, (start, end, name) in enumerate(rois):
    x1, y1 = start
    x2, y2 = end

    # Ensure ROI coordinates are within image bounds
    x1, x2 = max(0, x1), min(img_morph.shape[1], x2)
    y1, y2 = max(0, y1), min(img_morph.shape[0], y2)

    # Extract the ROI
    roi = img_morph[y1:y2, x1:x2]

    # Save the ROI as an image file with a descriptive name
    roi_filename = os.path.join(output_folder, f'{name}.jpg')
    cv2.imwrite(roi_filename, roi)

    # Perform OCR on the extracted ROI
    results = reader.readtext(roi, detail=1)  # detail=1 to get the bounding box, text, and confidence

    # Extract text and confidence
    if results:
        roi_text = ' '.join([result[1] for result in results])  # Combine all text results
        confidences = [result[2] for result in results]  # Extract confidence scores

        # Calculate the average accuracy for this ROI
        accuracy = np.mean(confidences) * 100 if confidences else 0
        print(f"{name}: {roi_text.strip()} (Accuracy: {accuracy:.2f}%)")

        # Update total confidences and texts
        total_confidences.extend(confidences)
        total_texts.append(roi_text.strip())

        # Store the extracted text in the dictionary
        if name in extracted_texts:
            extracted_texts[name] = roi_text.strip()
    else:
        print(f"{name}: No text detected.")

# Calculate and print the total average accuracy
total_accuracy = np.mean(total_confidences) * 100 if total_confidences else 0
print(f"Total Accuracy: {total_accuracy:.2f}%")

# Create a timestamp for the JSON filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_filename = f'extracted_texts_{timestamp}.json'
json_output_path = os.path.join(output_folder, json_filename)

# Save the JSON file in the desired format
with open(json_output_path, 'w') as json_file:
    json.dump(extracted_texts, json_file, indent=4)

# Display a ready message when the JSON file is ready
print(f"JSON file is ready and saved to {json_output_path}")
