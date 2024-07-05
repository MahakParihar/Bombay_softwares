import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))
    return resized_image.flatten()

# Load and preprocess all images
def load_images(data_dir):
    images = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                preprocessed_image = preprocess_image(image_path)
                images.append(preprocessed_image)
                labels.append(root.split(os.path.sep)[-1])  # Assuming the folder name is the label
    return np.array(images), np.array(labels)

# Function to extract histogram features
def extract_histogram_features(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

# Function to apply histogram equalization
def extract_histogram_equalization_features(image):
    equalized_image = cv2.equalizeHist(image)
    hist_eq = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    return hist_eq.flatten()

# Function to apply Canny edge detection
def extract_canny_features(image):
    edges = cv2.Canny(image, 100, 200)
    return edges.flatten()

# Main function to run the entire pipeline
def main():
    # Load images
    data_dir = 'C:/Users/mahakParihar/Downloads/dataset_1/dataset_full'  # Path to the dataset directory
    images, labels = load_images(data_dir)

    # Debugging: Print number of images and labels loaded
    print(f"Number of images loaded: {len(images)}")
    print(f"Number of labels loaded: {len(labels)}")

    # Ensure that images and labels are loaded correctly
    if len(images) == 0 or len(labels) == 0:
        print("No images or labels found. Check the dataset directory.")
        return

    # Extract features from images
    histogram_features = [extract_histogram_features(img) for img in images]
    histogram_eq_features = [extract_histogram_equalization_features(img) for img in images]
    canny_features = [extract_canny_features(img) for img in images]

    # Combine all features
    features = np.hstack((histogram_features, histogram_eq_features, canny_features))

    # Debugging: Print the shape of the features array
    print(f"Shape of features array: {features.shape}")

    # Apply PCA
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(features)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=0.2, random_state=42)

    # Train the classifier
    clf = SVC()
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model
    model_save_path = 'C:/Users/mahakParihar/OneDrive/Desktop/Project/model.pkl'
    joblib.dump(clf, model_save_path)
    print(f"Model saved successfully at: {model_save_path}")

    return clf  # Return the trained model if needed

if __name__ == "__main__":
    main()
