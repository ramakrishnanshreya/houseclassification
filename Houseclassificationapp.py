import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import shap
from io import BytesIO
import zipfile
import os
import pandas as pd
import tempfile
import matplotlib.pyplot as plt

# Load the model and labels
model = tf.keras.models.load_model('keras_model.h5')
with open('labels.txt', 'r') as file:
    class_labels = file.read().splitlines()

# Define image classification function
def classify_image(img_array):
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    class_label = class_labels[class_index]
    return class_label, confidence

# Define SHAP explanation function
def explain_image(img_array):
    # Ensure that the image array has the correct shape
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Create a SHAP explainer with a suitable masker
    masker = shap.maskers.Image("inpaint_telea", img_array.shape[1:4])
    explainer = shap.Explainer(model, masker)

    # Compute SHAP values
    shap_values = explainer(img_array)

    return shap_values

# Create Streamlit app
st.title("House Classification App")

uploaded_file = st.file_uploader("Choose a zip file...", type=["zip"])

if uploaded_file is not None:
    all_results = []
    all_shap_values = []

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        file_names = zip_ref.namelist()

        for file_name in file_names:
            # Read image from zip file
            img_data = zip_ref.read(file_name)
            img = Image.open(BytesIO(img_data)).resize((224, 224))
            
            # Normalize image values to [0, 1]
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Perform inference
            class_label, confidence = classify_image(img_array)

            # Explain the prediction using SHAP
            shap_values = explain_image(img_array)
            all_shap_values.append(shap_values)

            # Create a DataFrame for results
            data = {'Original Filename': [file_name], 'Classified as': [class_label], 'Confidence': [confidence]}
            all_results.append(data)

    # Display one contributing image for each class
    for i, label in enumerate(class_labels):
        st.write(f"Top Contributing Image for Class {label}")
        top_index = np.argmax(np.sum(np.abs(all_shap_values[i].values), axis=(1, 2, 3)))

        # Plot the image
        plt.imshow(img_array[0])
        plt.title(f"Contribution: {np.sum(np.abs(all_shap_values[i].values[0][top_index])):.4f}")
        st.pyplot()

    # Convert class labels to list of strings
    class_labels = list(map(str, class_labels))

    # Create a DataFrame for all results
    df_all_results = pd.DataFrame(all_results)

    # Create a zip file with individual classification results
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_results_path = os.path.join(temp_dir, "classification_results.zip")
        with zipfile.ZipFile(zip_results_path, "w") as zipf:
            for idx, data in enumerate(all_results):
                df = pd.DataFrame(data)
                csv_bytes = BytesIO()
                df.to_csv(csv_bytes, index=False)
                zipf.writestr(f"classification_results_{file_names[idx]}.csv", csv_bytes.getvalue())








           











