import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")

model = tf.keras.models.load_model("models/bird_drone_model.h5")

st.title("🛰️ Bird vs Drone Classifier")

file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    img = Image.open(file).resize((224,224))
    st.image(img, caption="Uploaded Image")

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    if pred[0][0] > 0.5:
        st.error("🚁 Drone Detected")
    else:
        st.success("🐦 Bird Detected")