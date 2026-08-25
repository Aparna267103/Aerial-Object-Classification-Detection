import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bird vs Drone Classifier",
    page_icon="🛰️",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title(
    "🛰️ Bird vs Drone Classifier"
)

st.write(
    "Upload an aerial image to classify it as Bird or Drone."
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "models/transfer_learning_best.keras"
    )


model = load_model()


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

file = st.file_uploader(
    "Upload Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if file is not None:

    # Open image
    img = Image.open(
        file
    ).convert("RGB")


    # Display image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )


    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    resized_img = img.resize(
        (224, 224)
    )


    img_array = np.array(
        resized_img
    ).astype(
        np.float32
    )


    # MobileNetV2 preprocessing
    img_array = preprocess_input(
        img_array
    )


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )


    probability = float(
        prediction[0][0]
    )


    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    if probability > 0.5:

        label = "Drone"

        confidence = probability

        st.error(
            "🚁 Drone Detected"
        )

    else:

        label = "Bird"

        confidence = 1 - probability

        st.success(
            "🐦 Bird Detected"
        )


    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    st.subheader(
        "Prediction Confidence"
    )


    st.progress(
        confidence
    )


    st.write(
        f"**{confidence * 100:.2f}%**"
    )


    # --------------------------------------------------
    # Model Information
    # --------------------------------------------------

    st.info(
        "Model: MobileNetV2 Transfer Learning"
    )
