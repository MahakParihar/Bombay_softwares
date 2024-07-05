from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))
    return resized_image.flatten()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file:
            image = preprocess_image(file)
            prediction = model.predict([image])[0]
            return jsonify({'prediction': str(prediction)})
        else:
            return jsonify({'error': 'No file provided'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
