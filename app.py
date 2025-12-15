import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os
from PIL import Image

# -----------------------------
# Google Drive model download
# -----------------------------
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id=14zaQHuHJ0lleXvXNRfFSsnHUZfTFbmHk"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# Load CNN model
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Brain Tumor / Alzheimer Detection", layout="centered")

st.title("🧠 Brain Disease Detection")
st.write("Upload an MRI image for prediction")

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(img):
    img = img.resize((224, 224))   # must match training size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

# -----------------------------
# Display + Predict
# -----------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        prediction = predict_image(img)
        class_index = np.argmax(prediction)

        class_names = [
            "No Disease",
            "Alzheimer",
            "Brain Tumor",
            "Mild Cognitive Impairment"
        ]  # EDIT if needed

        st.success(f"Prediction: **{class_names[class_index]}**")
