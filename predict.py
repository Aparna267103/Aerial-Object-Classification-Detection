import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

tf.get_logger().setLevel("ERROR")
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bird_drone_model.h5"
DEFAULT_IMAGE = BASE_DIR / "dataset" / "test" / "bird" / "1e5479fa848be57b_jpg.rf.126c4dfb574482a7ff940570206491a3.jpg"


def predict_image(image_path: Path) -> str:
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict(img_array, verbose=0)

    return "Drone 🚁" if prediction[0][0] > 0.5 else "Bird 🐦"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a bird or drone image")
    parser.add_argument("image_path", nargs="?", default=str(DEFAULT_IMAGE), help="Path to the input image")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(predict_image(image_path))