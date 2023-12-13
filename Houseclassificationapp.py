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
def classify_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    class_label = class_labels[class_index]
    return class_label, confidence, img_array

# Define SHAP explanation function
def explain_image(img_array):
    # Create a SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(img_array)

    # Get class index and label
    class_index = np.argmax(model.predict(img_array)[0])
    class_label = class_labels[class_index]

    # Plot SHAP values
    shap_plot = shap.image_plot(shap_values, img_array, class_names=[class_label], show=False)
    return shap_plot

# Create Streamlit app
st.title("House Classification App")

uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Perform inference
    _, _, img_array = classify_image(uploaded_file)

    # Explain the prediction using SHAP
    shap_plot = explain_image(img_array)

    # Create a download button for the SHAP plot
    shap_bytes = BytesIO()
    shap_plot.savefig(shap_bytes, format='png')
    st.download_button(label="Download SHAP Plot", data=shap_bytes.getvalue(), file_name="shap_plot.png", key="shap_plot")

    # Create a DataFrame (replace this with your actual data)
    data = {'Original Filename': [uploaded_file.name], 'Classified as': ['Class not displayed'], 'Confidence': [0.0]}
    df = pd.DataFrame(data)

    # Create a download button for the CSV file
    csv_bytes = BytesIO()
    df.to_csv(csv_bytes, index=False)
    st.download_button(label="Download CSV", data=csv_bytes.getvalue(), file_name="classification_results.csv", key="csv_results")

    # Create a download button for the zip file
    with zipfile.ZipFile("classification_results.zip", "w") as zipf:
        zipf.writestr("classification_results.csv", df.to_csv(index=False))

    st.download_button(label="Download Zip", data=None, file_name="classification_results.zip", key="zip_results")

    # Close the SHAP plot to free up resources
    st.pyplot(shap_plot)



