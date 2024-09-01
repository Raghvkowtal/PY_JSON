import cv2
import pytesseract
import numpy as np
import os
import json
from datetime import datetime

# Check if Tesseract OCR is installed
try:
    pytesseract.get_tesseract_version()
except OSError:
    raise EnvironmentError("Tesseract OCR is not installed or not found.")

# Define the file path
file_path = 'D:\\ftest - Copy\\image .jpg'

# Load the image using OpenCV
img = cv2.imread(file_path)

# Check if image was loaded successfully
if img is None:
    print(f"Error: The image file at {file_path} could not be opened. Please check the file path.")
    exit(1)

# Resize the image
img = cv2.resize(img, (2000, 2000))  # Adjust as needed

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
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

# Define ROIs
rois = [
    [(513, 461), (1194, 549), 'patient_full_name'],
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

# Initialize an empty dictionary to hold extracted texts
extracted_texts = {name: "" for _, _, name in rois}

# Process each ROI
for start, end, name in rois:
    x1, y1 = start
    x2, y2 = end

    # Ensure ROI coordinates are within image bounds
    x1, x2 = max(0, x1), min(img_morph.shape[1], x2)
    y1, y2 = max(0, y1), min(img_morph.shape[0], y2)

    # Extract and save the ROI
    roi = img_morph[y1:y2, x1:x2]
    roi_filename = os.path.join(output_folder, f'{name}.jpg')
    cv2.imwrite(roi_filename, roi)

    # Perform OCR with Tesseract
    roi_text = pytesseract.image_to_string(roi, config='--psm 6')  # Experiment with different psm values

    # Update and print results
    extracted_texts[name] = roi_text.strip()
    print(f"{name}: {roi_text.strip()}")

# Save the extracted texts to JSON
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_filename = f'extracted_texts_{timestamp}.json'
json_output_path = os.path.join(output_folder, json_filename)

with open(json_output_path, 'w') as json_file:
    json.dump(extracted_texts, json_file, indent=4)

print(f"JSON file is ready and saved to {json_output_path}")
