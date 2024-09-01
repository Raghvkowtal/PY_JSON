from flask import Flask, request, render_template
from PIL import Image
import cv2
import numpy as np
import pytesseract
from easyocr import Reader
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def extract_roi(image_path, x1, y1, x2, y2):
    """Extracts the region of interest (ROI) from the image."""
    image = Image.open(image_path)
    roi_image = image.crop((x1, y1, x2, y2))
    roi_image_path = image_path.replace('.jpg', '_roi.jpg')
    roi_image.save(roi_image_path)
    return roi_image_path

def preprocess_image(image_path):
    """Preprocesses an image for better OCR results."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    processed_image_path = image_path.replace('.jpg', '_processed.jpg')
    cv2.imwrite(processed_image_path, thresh)
    return processed_image_path

def ocr_scan(image_path):
    """Extracts text from an image using both EasyOCR and Tesseract OCR."""
    reader = Reader(['en'])
    # EasyOCR
    easyocr_results = reader.readtext(image_path)
    easyocr_text = " ".join([text[1] for text in easyocr_results])
    
    # Tesseract OCR
    tesseract_text = pytesseract.image_to_string(image_path)
    
    return easyocr_text, tesseract_text

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or not all(k in request.form for k in ('x1', 'y1', 'x2', 'y2')):
        return "Missing file or ROI coordinates", 400

    file = request.files['file']
    x1, y1, x2, y2 = int(request.form['x1']), int(request.form['y1']), int(request.form['x2']), int(request.form['y2'])
    
    if file.filename == '':
        return "No selected file", 400

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    roi_image_path = extract_roi(file_path, x1, y1, x2, y2)
    processed_image_path = preprocess_image(roi_image_path)
    easyocr_text, tesseract_text = ocr_scan(processed_image_path)

    return render_template('result.html', easyocr_text=easyocr_text, tesseract_text=tesseract_text)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
