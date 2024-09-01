# from flask import Flask, request, render_template, redirect, url_for, send_from_directory
# import easyocr
# import pytesseract
# import os
# import tempfile
# import fitz
# from PIL import Image, ImageEnhance, ImageFilter, ImageOps
# import io
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torchvision import models
# from torch.utils.data import Dataset, DataLoader
# import json

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['LABELS_FILE'] = 'labels.json'  # File to store labels

# # Check if GPU is available and log the status
# if torch.cuda.is_available():
#     print("CUDA is available. Using GPU.")
# else:
#     print("CUDA is not available. Using CPU.")

# # Initialize the OCR reader
# reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# # Ensure the upload folder exists
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# # Ensure the labels file exists
# if not os.path.exists(app.config['LABELS_FILE']):
#     with open(app.config['LABELS_FILE'], 'w') as f:
#         json.dump({}, f)

# # Path to Tesseract executable (update this path if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update to your Tesseract path

# # Load pre-trained ResNet model
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust this if you have different classes
# model.eval()

# # Image transformation
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files or 'label' not in request.form or not all(k in request.form for k in ('x1', 'y1', 'x2', 'y2')):
#         return redirect(request.url)
    
#     file = request.files['file']
#     label = request.form['label']
    
#     # Extract ROI coordinates
#     x1 = int(request.form['x1'])
#     y1 = int(request.form['y1'])
#     x2 = int(request.form['x2'])
#     y2 = int(request.form['y2'])

#     if file.filename == '':
#         return redirect(request.url)

#     if file:
#         filename = file.filename
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Save the label associated with this image
#         save_label(filename, label)

#         # Extract the ROI from the image
#         roi_image_path = extract_roi(file_path, x1, y1, x2, y2)

#         recognized_text = ocr_scan(roi_image_path)
#         return render_template('result.html', text=recognized_text, file_url=url_for('uploaded_file', filename=filename))

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/classify', methods=['POST'])
# def classify_image():
#     if 'file' not in request.files or not all(k in request.form for k in ('x1', 'y1', 'x2', 'y2')):
#         return redirect(request.url)
#     file = request.files['file']
    
#     # Extract ROI coordinates
#     x1 = int(request.form['x1'])
#     y1 = int(request.form['y1'])
#     x2 = int(request.form['x2'])
#     y2 = int(request.form['y2'])

#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         filename = file.filename
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Extract the ROI from the image
#         roi_image_path = extract_roi(file_path, x1, y1, x2, y2)

#         classification_result = classify(roi_image_path)
#         return render_template('classification_result.html', result=classification_result, file_url=url_for('uploaded_file', filename=filename))

# @app.route('/train', methods=['POST'])
# def train_model():
#     num_epochs = int(request.form.get('epochs', 5))
#     learning_rate = float(request.form.get('learning_rate', 0.001))

#     # Initialize dataset and DataLoader
#     train_dataset = CustomDataset(image_folder=app.config['UPLOAD_FOLDER'], transform=preprocess)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#     # Define the optimizer and loss function
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     # Training loop
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             if torch.cuda.is_available():
#                 inputs, labels = inputs.to('cuda'), labels.to('cuda')
#                 model.to('cuda')

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

#     # Save the model
#     torch.save(model.state_dict(), 'trained_model.pth')
#     return "Training complete. Model saved as 'trained_model.pth'."

# def classify(image_path):
#     input_image = Image.open(image_path).convert('RGB')
#     input_tensor = preprocess(input_image)
#     input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

#     if torch.cuda.is_available():
#         input_batch = input_batch.to('cuda')
#         model.to('cuda')

#     with torch.no_grad():
#         output = model(input_batch)
#     # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
#     probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
#     # Get the top 5 probabilities and class indices
#     top5_prob, top5_catid = torch.topk(probabilities, 5)
    
#     # Load the class labels
#     with open("imagenet_classes.txt") as f:
#         categories = [s.strip() for s in f.readlines()]
    
#     results = [(categories[catid], prob.item()) for catid, prob in zip(top5_catid, top5_prob)]
#     return results

# def ocr_scan(file_path):
#     if file_path.lower().endswith('.pdf'):
#         # PDF file detected, extract images and perform OCR
#         image_paths = extract_images_from_pdf(file_path)
#         recognized_text = ""
#         for img_path in image_paths:
#             recognized_text += ocr_scan(img_path) + "\n\n"
#         return recognized_text.strip()
#     else:
#         # Image file detected, perform OCR with preprocessing
#         preprocessed_image_path = preprocess_image(file_path)
#         easyocr_text = perform_easyocr(preprocessed_image_path)
#         tesseract_text = perform_tesseract_ocr(preprocessed_image_path)
#         # Combine results from both OCR systems
#         recognized_text = f"EasyOCR: {easyocr_text}\nTesseract: {tesseract_text}"
#         return recognized_text

# def extract_roi(image_path, x1, y1, x2, y2):
#     image = Image.open(image_path)
#     roi_image = image.crop((x1, y1, x2, y2))
#     roi_image_path = f"{tempfile.mkdtemp()}/{os.path.basename(image_path)}_roi.jpg"
#     roi_image.save(roi_image_path)
#     return roi_image_path

# def preprocess_image(image_path):
#     image = Image.open(image_path)
#     # Convert image to grayscale
#     image = image.convert('L')
#     # Invert image colors for better OCR on dark text
#     image = ImageOps.invert(image)
#     # Enhance contrast
#     enhancer = ImageEnhance.Contrast(image)
#     image = enhancer.enhance(2)
#     # Apply a sharpening filter
#     image = image.filter(ImageFilter.SHARPEN)
#     # Denoise image
#     image = image.filter(ImageFilter.MedianFilter(size=3))
#     # Save the preprocessed image
#     preprocessed_image_path = f"{tempfile.mkdtemp()}/{os.path.basename(image_path)}_preprocessed.jpg"
#     image.save(preprocessed_image_path, "JPEG")
#     return preprocessed_image_path

# def perform_easyocr(image_path):
#     result = reader.readtext(image_path)
#     recognized_text = " ".join(elem[1] for elem in result)
#     return recognized_text

# def perform_tesseract_ocr(image_path):
#     text = pytesseract.image_to_string(image_path)
#     return text

# def extract_images_from_pdf(pdf_path):
#     image_paths = []
#     with fitz.open(pdf_path) as doc:
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
#             image_list = page.get_images(full=True)
#             for img_index, img in enumerate(image_list):
#                 xref = img[0]
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image = Image.open(io.BytesIO(image_bytes))
#                 if image.mode == 'RGBA':
#                     image = image.convert('RGB')  # Convert RGBA to RGB
#                 image_path = f"{tempfile.mkdtemp()}/{os.path.basename(pdf_path)}_page{page_num}_img{img_index}.jpg"
#                 image.save(image_path, "JPEG")
#                 image_paths.append(image_path)
#     return image_paths

# class CustomDataset(Dataset):
#     def __init__(self, image_folder, transform=None):
#         self.image_folder = image_folder
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
#         self.labels = self.load_labels()

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.image_folder, self.image_files[idx])
#         image = Image.open(img_name).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         label = self.labels.get(self.image_files[idx], 0)  # Default label if not found
#         return image, label

#     def load_labels(self):
#         with open(app.config['LABELS_FILE'], 'r') as f:
#             labels = json.load(f)
#         return labels

# def save_label(filename, label):
#     with open(app.config['LABELS_FILE'], 'r') as f:
#         labels = json.load(f)

#     labels[filename] = int(label)  # Assuming labels are integers

#     with open(app.config['LABELS_FILE'], 'w') as f:
#         json.dump(labels, f)

# if __name__ == '__main__':
#     app.run(debug=True)

import cv2
import easyocr
import os

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Use the appropriate language(s)

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

# Create a folder to save the ROIs if it doesn't exist
output_folder = 'D:\\Extracted_ROIs'
os.makedirs(output_folder, exist_ok=True)

# Initialize an empty dictionary to hold extracted texts
extracted_texts = {}

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

    # Store the extracted text in the dictionary
    extracted_texts[name] = roi_text.strip()

    # Display the ROI and the extracted text
    cv2.imshow(f'{name}', roi)
    print(f"{name}: {roi_text.strip()}")

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, you can save the extracted texts to a file
with open(os.path.join(output_folder, 'extracted_texts.txt'), 'w') as f:
    for name, text in extracted_texts.items():
        f.write(f"{name}: {text}\n")
