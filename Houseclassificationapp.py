import os
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
def classify_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            class_label, confidence = classify_image(image_path)
            results.append({
                'filename': filename,
                'class_label': class_label,
                'confidence': confidence
            })
    return results

# Create Streamlit app
st.title("Image Classification App")

uploaded_folder = st.folder_uploader("Choose a folder...", type=["jpg", "jpeg", "png"])

if uploaded_folder is not None:
    st.write(f"Classifying images in folder: {uploaded_folder}")

    # Perform inference on all images in the folder
    results = classify_folder(uploaded_folder)

    # Display results
    for result in results:
        st.write(f"Filename: {result['filename']}")
        st.write(f"Predicted Class: {result['class_label']}")
        st.write(f"Confidence Score: {result['confidence']:.2%}")
        st.write("----")

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Add a download button for the results
    st.download_button(
        label="Download Results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='image_classification_results.csv',
        key='download_results_button'
    )

    # Perform inference
    class_label, confidence = classify_image(uploaded_file)
    st.write(f"Predicted Class: {class_label}")
    st.write(f"Confidence Score: {confidence:.2%}")
