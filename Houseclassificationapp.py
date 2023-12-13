import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import shap
from io import BytesIO
import zipfile
import os
import pandas as pd

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
    # Create a SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(img_array)
    return shap_values

# Create Streamlit app
st.title("House Classification App")

uploaded_file = st.file_uploader("Choose a zip file...", type=["zip"])

if uploaded_file is not None:
    all_shap_values = []

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        file_names = zip_ref.namelist()

        for file_name in file_names:
            # Read image from zip file
            img_data = zip_ref.read(file_name)
            img = Image.open(BytesIO(img_data)).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Perform inference
            class_label, confidence = classify_image(img_array)

            # Explain the prediction using SHAP
            shap_values = explain_image(img_array)
            all_shap_values.append(shap_values)

            # Create a DataFrame for results
            data = {'Original Filename': [file_name], 'Classified as': [class_label], 'Confidence': [confidence]}
            df = pd.DataFrame(data)

            # Create a download button for the CSV file
            csv_bytes = BytesIO()
            df.to_csv(csv_bytes, index=False)
            st.download_button(label=f"Download CSV ({file_name})", data=csv_bytes.getvalue(), file_name=f"classification_results_{file_name}.csv", key=f"csv_results_{file_name}")

    # Aggregate SHAP values
    all_shap_values = np.array(all_shap_values)
    mean_shap_values = np.mean(all_shap_values, axis=0)

    # Plot aggregated SHAP values
    st.write("Aggregated SHAP Plot for All Images")
    shap.image_plot(mean_shap_values, img_array, class_names=class_labels, show=True)




