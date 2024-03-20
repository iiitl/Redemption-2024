# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:14:05 2024

@author: Asus
"""

from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("D:/AI-ML/NeuralNetworks/ConvolutedNets/Projects/CatDogClassifier/CatDogClassifier.h5")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_class(image):
    processed_image = cv2.resize(image, (256, 256))
    processed_image = processed_image.astype('float32') / 255
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class = 'Dog' if predictions > 0.5 else 'Cat'
    return predicted_class

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if image_file and allowed_file(image_file.filename):
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            predicted_class = predict_class(image)
            return jsonify({'class': predicted_class})
        else:
            return jsonify({'error': 'Unsupported image format'}), 415
    else:
        return jsonify({'error': 'Method not allowed'}), 405

@app.route('/', methods=['GET'])
def index():
    # all the FrontEnd
    return "<h1>Cat or Dog Classifier (Flask)</h1>"

if __name__ == '__main__':
    app.run(debug=True)
