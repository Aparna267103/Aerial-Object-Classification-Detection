import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")


# --------------------------------------------------
# 1. Model Path
# --------------------------------------------------

MODEL_PATH = "models/transfer_learning_best.keras"


# --------------------------------------------------
# 2. Load Trained Model to memory
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


print("Model loaded successfully.")


# --------------------------------------------------
# 3. Prediction Function
# --------------------------------------------------

def predict_image(image_path):

    # Load image
    img = Image.open(
        image_path
    ).convert("RGB")


    # Resize image
    img = img.resize(
        (224, 224)
    )


    # Convert to NumPy array
    img_array = np.array(
        img
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


    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )


    probability = float(
        prediction[0][0]
    )


    # Classification
    if probability > 0.5:

        label = "Drone"

        confidence = probability

    else:

        label = "Bird"

        confidence = 1 - probability


    return label, confidence


# --------------------------------------------------
# 4. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    ).strip().strip('"')


    if not os.path.exists(image_path):

        print(
            "Error: The path does not exist. Enter the path to an image file."
        )

    elif not os.path.isfile(image_path):

        print(
            "Error: The path is a directory. Enter a specific image file, "
            "for example: dataset\\test\\bird\\image.jpg"
        )

    else:

        label, confidence = predict_image(
            image_path
        )


        print("\nPrediction Result")
        print("------------------")

        print(
            "Class:",
            label
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )
