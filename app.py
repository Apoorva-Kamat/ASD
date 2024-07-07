from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model = load_model('autism_detection_model.h5')

# Set upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to the target size
    resized_image = cv2.resize(gray_image, (128, 128))
    # Normalize the image
    normalized_image = resized_image.astype('float32') / 255.0
    # Reshape to match model input shape
    reshaped_image = normalized_image.reshape(1, 128, 128, 1)
    return reshaped_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    # Decode base64 image
    image_data = base64.b64decode(data)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Read as color image to match the preprocess function
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    result = 'Autism' if prediction[0][1] < 0.5 else 'No Autism'
    
    return jsonify(result=result)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify(result='No file part')
    file = request.files['file']
    if file.filename == '':
        return jsonify(result='No selected file')
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Process the uploaded image
        image = cv2.imread(file_path)
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        result = 'Autism' if prediction[0][1] > 0.5 else 'No Autism'
        
        return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
