from flask import Flask, request, jsonify
#pip install Flask
import os
import cv2
#pip install opencv-python
import numpy as np
#pip install numpy
from sklearn.neighbors import KNeighborsClassifier
#pip install
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import glob

app = Flask(__name__)

# Folder paths to train and test data
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'

# Initialize variables for the classifier
knn = None
label_encoder = None

# Function to load images from a folder and preprocess them
def load_images_from_folder(folder):
    images = []
    labels = []
    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize image
                img = img.flatten()  # Flatten the image for KNN (turn into 1D array)
                images.append(img)
                labels.append(class_name)
    
    # Convert the lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Check that images is 2D and labels is 1D
    print(f"Loaded {images.shape[0]} images with shape {images.shape} and {len(labels)} labels")

    return images, labels


# Load and train the KNN model
def train_knn():
    global knn, label_encoder

    # Load training images and labels
    train_images, train_labels = load_images_from_folder(TRAIN_FOLDER)

    # Encode labels into integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)

    # Ensure train_images is a 2D array and train_labels_encoded is 1D
    print(f"Training images shape: {train_images.shape}, Labels shape: {train_labels_encoded.shape}")

    # Train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_images, train_labels_encoded)
    image = cv2.resize(image, (64, 64))  # Resize to match training images
    image = image.flatten()  # Flatten for classification
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from the request
        file = request.files['image']
        image = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict using the trained KNN model
        prediction = knn.predict([processed_image])
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Return the predicted label
        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Train the KNN model at startup
    train_knn()
    app.run(debug=True)

    #{0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9:'J', 10: 'K', 11: 'L', 12: 'M', 13:
     #               'N', 14:'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19:'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24:'Y', 25: 'Z'}