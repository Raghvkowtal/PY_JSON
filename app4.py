import cv2
import easyocr
import os
import numpy as np
import json
import torch
from datetime import datetime

# Initialize EasyOCR reader with GPU support if available
reader = easyocr.Reader(['en'], gpu=True if torch.cuda.is_available() else False)

# Load the image
file_path = 'D:\\Clone\\form_image_1.jpg'
img = cv2.imread(file_path)

if img is None:
    print(f"Error: The image file at {file_path} could not be opened. Please check the file path.")
    exit(1)

# Resize image (Optional: based on the original image resolution)
# img = cv2.resize(img, (2000, 2000))

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Increase contrast (optional: skip if the original image has good contrast)
img_hist_eq = cv2.equalizeHist(img_gray)

# Perform Gaussian blur to reduce noise (adjust kernel size as needed)
img_blur = cv2.GaussianBlur(img_hist_eq, (3, 3), 0)

# Perform adaptive thresholding (optional: use if needed)
img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

# Perform morphological operations (optional: refine based on image characteristics)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

# Visualize ROI for debugging
debug_img = img.copy()
for (start, end, name) in rois:
    cv2.rectangle(debug_img, start, end, (0, 255, 0), 2)
cv2.imshow("ROIs", debug_img)
cv2.waitKey(0)

# Extract and OCR each ROI
extracted_texts = {}
total_confidences = []

for start, end, name in rois:
    roi = img_morph[start[1]:end[1], start[0]:end[0]]
    results = reader.readtext(roi, detail=1)
    
    if not results:
        print(f"{name}: No text detected.")
        extracted_texts[name] = {"extracted_text": ""}
        continue
    
    roi_text = ' '.join([result[1] for result in results])
    confidences = [result[2] for result in results]
    accuracy = np.mean(confidences) * 100 if confidences else 0
    print(f"{name}: {roi_text.strip()} (Accuracy: {accuracy:.2f}%)")
    total_confidences.extend(confidences)
    extracted_texts[name] = {"extracted_text": roi_text.strip()}

total_accuracy = np.mean(total_confidences) * 100 if total_confidences else 0
print(f"Total Accuracy: {total_accuracy:.2f}%")

# Save extracted text to JSON
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_output_path = os.path.join('Extracted_Json', f'extracted_texts_{timestamp}.json')
with open(json_output_path, 'w') as json_file:
    json.dump(extracted_texts, json_file, indent=4)

print(f"JSON file is ready and saved to {json_output_path}")
