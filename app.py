import os
import cv2
import numpy as np
#import face_recognition
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LogisticRegression
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model
model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
logistic_model = LogisticRegression()
x_train, y_train = [], []

@app.route('/')
def index():
    """Render home page"""
    return render_template("index.html")

@app.route('/capture', methods=['POST'])
def capture_image():
    """Capture image with label"""
    label = request.form.get('label')
    if 'image' not in request.files or not label:
        return jsonify({'error': 'Image or label missing'}), 400

    image_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    # Process image
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (120, 120))
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB) / 255.0
        sample = gray_face.flatten()
        x_train.append(sample)
        y_train.append(label)

    return jsonify({'message': f'Captured {label} image successfully'})

@app.route('/train', methods=['POST'])
def train_model():
    """Train Logistic Regression model"""
    if not x_train or not y_train:
        return jsonify({'error': 'No training data available'}), 400

    logistic_model.fit(x_train, y_train)
    return jsonify({'message': 'Model trained successfully'})

@app.route('/detect', methods=['POST'])
def detect_faces():
    """Detect faces in an uploaded image"""
    if not hasattr(logistic_model, "coef_"):
        return jsonify({'error': 'Model not trained'}), 400

    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    # Read and process the image
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
    results = []

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (120, 120))
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB) / 255.0
        sample = gray_face.flatten().reshape(1, -1)

        # Make prediction
        prediction = logistic_model.predict(sample)
        label = prediction[0]
        results.append({'label': label, 'bbox': [x, y, w, h]})

    return jsonify({'faces_detected': len(results), 'results': results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
