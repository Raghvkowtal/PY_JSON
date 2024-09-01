import cv2
import easyocr
import os
import torch
import json
from difflib import SequenceMatcher

# Check if GPU is available
use_gpu = torch.cuda.is_available()
print("CUDA available: ", use_gpu)

# Initialize EasyOCR reader with GPU support if available
reader = easyocr.Reader(['en'], gpu=use_gpu)

# Load the image
img = cv2.imread('D:\\Clone\\form image 1 .jpg')

# Resize the image if necessary
img = cv2.resize(img, (1000, 1000))

# List of ROIs with coordinates and descriptive names
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

# Ground truth data for accuracy calculation
ground_truth = {
    "patient_full_name": "John Doe",
    "date_of_birth": "01/01/1980",
    "email": "johndoe@example.com",
    "home_address": "1234 Main St",
    "apt": "Apt 101",
    "city": "Anytown",
    "state": "CA",
    "zipcode": "12345",
    "phone1": "(123) 456-7890",
    "emergency_contact": "Jane Doe",
    "phone2": "(987) 654-3210",
    "primary_insurance": "InsuranceCo",
    "policy": "123456789",
    "secondary_insurance": "SecondCo",
    "policy_number": "987654321",
    "secondary_phone": "(555) 555-5555"
}

# Create a folder to save the ROIs if it doesn't exist
output_folder = 'D:\\Extracted_ROIs'
os.makedirs(output_folder, exist_ok=True)

# Initialize an empty dictionary to hold extracted texts (without accuracy)
extracted_texts = {}

# Function to calculate similarity between two strings
def calculate_accuracy(ground_truth_text, extracted_text):
    return SequenceMatcher(None, ground_truth_text, extracted_text).ratio()

# Loop through each ROI
for i, (start, end, name) in enumerate(rois):
    x1, y1 = start
    x2, y2 = end

    # Extract the ROI
    roi = img[y1:y2, x1:x2]

    # Save the ROI as an image file with a descriptive name
    roi_filename = os.path.join(output_folder, f'{name}.jpg')
    cv2.imwrite(roi_filename, roi)

    # Perform OCR on the extracted ROI
    results = reader.readtext(roi)
    roi_text = ' '.join([result[1] for result in results])  # Combine all text results

    # Calculate accuracy for this field
    accuracy = calculate_accuracy(ground_truth[name], roi_text.strip())

    # Store the extracted text in the dictionary without accuracy
    extracted_texts[name] = {
        "extracted_text": roi_text.strip()
    }

    # Display the extracted text and accuracy in the console
    cv2.imshow(f'{name}', roi)
    print(f"{name}: {roi_text.strip()} (Accuracy: {accuracy:.2%})")

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the extracted texts (without accuracy) to a JSON file
json_output_path = os.path.join(output_folder, 'extracted_texts.json')
with open(json_output_path, 'w') as f:
    json.dump(extracted_texts, f, indent=4)

# Calculate and print overall accuracy
overall_accuracy = sum(calculate_accuracy(ground_truth[name], extracted_texts[name]['extracted_text']) for name in extracted_texts) / len(extracted_texts)
print(f"\nOverall OCR Accuracy: {overall_accuracy:.2f}%")

print(f"OCR processing completed. Extracted texts saved to '{json_output_path}'.")
