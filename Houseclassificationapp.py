import os
import zipfile
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model and labels
model = tf.keras.models.load_model('keras_model.h5')
with open('labels.txt', 'r') as file:
    class_labels = file.read().splitlines()

# Define image classification function
def classify_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    class_label = class_labels[class_index]
    return class_label, confidence

# Function to classify all images in a folder
def classify_images(image_paths, output_folder):
    results = []
    for image_path in image_paths:
        # Check if the path is a file and ends with a supported image extension
        if os.path.isfile(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                class_label, confidence = classify_image(image_path)
                results.append({
                    'filename': os.path.basename(image_path),
                    'class_label': class_label,
                    'confidence': confidence
                })

                # Create a folder for each class
                class_folder = os.path.join(output_folder, class_label)
                os.makedirs(class_folder, exist_ok=True)

                # Move the image to the corresponding class folder
                new_path = os.path.join(class_folder, os.path.basename(image_path))
                os.rename(image_path, new_path)

            except Exception as e:
                st.warning(f"Error classifying image {image_path}: {e}")
        else:
            st.warning(f"Skipping non-image file: {image_path}")

    return results

# Create Streamlit app
st.title("Image Classification App")

uploaded_zip = st.file_uploader("Upload a zip file containing images", type=["zip"])

if uploaded_zip is not None:
    # Create an output folder to organize images by class
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        extraction_path = "temp_images"
        zip_ref.extractall(extraction_path)

    st.write(f"Classifying and organizing images in folder: {extraction_path}")

    # Get the list of image paths in the extracted folder
    image_paths = [os.path.join(extraction_path, filename) for filename in os.listdir(extraction_path)]

    # Perform inference on all images in the folder and organize by class
    results = classify_images(image_paths, output_folder)

    # Display results
    for result in results:
        st.write(f"Original Filename: {result['filename']}")
        st.write(f"Classified as: {result['class_label']} with Confidence: {result['confidence']:.2%}")
        st.write("----")

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Add a download button for the CSV file with image names and classes
    st.download_button(
        label="Download Results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='image_classification_results.csv',
        key='download_results_button'
    )

    # Create a zip file containing the organized images
    organized_zip_path = "organized_images.zip"
    with zipfile.ZipFile(organized_zip_path, 'w') as zip_output:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                zip_output.write(file_path, os.path.relpath(file_path, output_folder))

    # Add a download button for the zip file with organized images
    st.download_button(
        label="Download Organized Images",
        data=open(organized_zip_path, "rb").read(),
        file_name='organized_images.zip',
        key='download_organized_button'
    )










           











