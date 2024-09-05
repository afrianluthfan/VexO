import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pytesseract
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warnings, only errors will be shown
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use only CPU
from keras.api.applications import Xception
from keras.api.models import Sequential, load_model
from keras.api.layers import Dense, Input
from keras.api.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split

# Define paths to your data
real_image_dir = './real'
fake_image_dir = './fake'

# Load the Xception model, pretrained on ImageNet
xception_model = Xception(weights='imagenet', include_top=False, pooling='avg')

def load_and_preprocess_image(path=None, image_array=None):
    if path:
        img = cv2.imread(path)
        x = cv2.resize(img, (299, 299))
    elif image_array is not None:
        x = cv2.resize(image_array, (299, 299))
    else:
        raise ValueError("Either path or image_array must be provided.")
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_features(image_array):
    img = load_and_preprocess_image(image_array=image_array)
    features = xception_model.predict(img)
    return features

def ocr_check(image):
    text = pytesseract.image_to_string(image)
    text = text.lower()  # Convert text to lowercase for easier matching
    if "faceswap" in text or "manycam" in text:
        return True
    return False

def load_images_from_directory(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png:Zone.Identifier', '.jpg:Zone.Identifier')):
            continue
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            features = extract_features(img)
            features = np.squeeze(features)
            images.append(features)
            label = 1 if directory == real_image_dir else 0
            labels.append(label)
    return np.array(images), np.array(labels)

def train_model():
    real_images, real_labels = load_images_from_directory('./real')
    fake_images, fake_labels = load_images_from_directory('./fake')
    images = np.concatenate((real_images, fake_images))
    labels = np.concatenate((real_labels, fake_labels))
    if images.size == 0 or labels.size == 0:
        print("No images loaded. Please check your directories.")
        return
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    model = Sequential([
        Input(shape=(2048,)),  # Adjust this to 2048 to match the Xception output
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)
    model.save('image_classifier_model.keras')
    print("Model training complete and saved.")

def update_model(features, correct_label):
    try:
        model = load_model('image_classifier_model.keras')
        X_train = np.expand_dims(features, axis=0)
        y_train = np.array([correct_label])
        model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1)
        model.save('image_classifier_model.keras')
        print("Model updated and saved with the corrected label: ", correct_label)
    except Exception as e:
        print("Error loading model: ", e)


def process_image_test(model, img_path):
    if not os.path.exists(img_path):
        print("Error: The file does not exist.")
        return
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Unable to load the image.")
        return

    # Perform OCR check before extracting features
    if ocr_check(image):
        print("Warning: Detected text indicating 'faceswap' or 'manycam'. This image may be manipulated.")
        return

        # Extract features and make a prediction
    features = extract_features(image)
    features = np.squeeze(features)  # Remove extra dimension
    score = model.predict(np.expand_dims(features, axis=0))[0][0]
    print(f"Prediction confidence score: {score:.4f}")

    # Output the result
    if score >= 0.5:
        print("Gambar ini diklasifikasikan sebagai asli.")
        predicted_label = 1
    else:
        print("Gambar ini diklasifikasikan sebagai palsu.")
        predicted_label = 0

    # Ask for user confirmation
    correct = input("Apakah prediksi benar? (y/t): ").strip().lower()
    if correct == 't':
        correct_label = 0 if predicted_label == 1 else 1
        update_model(features, correct_label)
    else:
        print("No update needed.")

def test_model():
    """Test the model using an image provided by the user."""
    # Load the trained model
    try:
        model = load_model('image_classifier_model.keras')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get the image path from the user
    img_path = input("Enter the image path to test: ").strip()

    if not os.path.exists(img_path):
        print(f"Error: File {img_path} tidak ditemukan.")
        return

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Tidak dapat memuat gambar. Silakan cek path file dan keberadaan file tersebut.")
        return

    process_image_test(model, img_path)

def test_multiple_images():
    try:
        model = load_model('image_classifier_model.keras')
        file_paths = choose_multiple_images()
        if not file_paths:
            print("Tidak ada gambar yang dipilih.")
            return
        for img_path in file_paths:
            if not os.path.exists(img_path):
                print("Error: File tidak ditemukan.")
                continue
            image = cv2.imread(img_path)
            if image is None:
                print("Error: Tidak dapat memuat gambar.")
                continue
            features = extract_features(image)
            features = np.squeeze(features)
            score = model.predict(np.expand_dims(features, axis=0))[0][0]
            print("Image: {} - Prediction confidence score: {:.4f}".format(img_path, score))
            if score >= 0.5:
                print("The image is classified as real.")
            else:
                print("The image is classified as fake.")
    except Exception as e:
        print("Error loading model: ", e)

def choose_multiple_images():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title='Select Images', filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return root.tk.splitlist(file_paths)

def main_menu():
    while True:
        print("\nChoose an option:")
        print("1. Train the model")
        print("2. Test a single image")
        print("3. Test multiple images")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")
        if choice == '1':
            train_model()
        elif choice == '2':
            test_model()
        elif choice == '3':
            test_multiple_images()
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a valid option (1-4).")

if __name__ == "__main__":
    main_menu()