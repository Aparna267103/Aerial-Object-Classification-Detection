import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks


logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Data Augmentation
# --------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# --------------------------------------------------
# 3. Load Training Data
# --------------------------------------------------

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)


# --------------------------------------------------
# 4. Load Validation Data
# --------------------------------------------------

valid_data = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)


print("Class Mapping:", train_data.class_indices)


# --------------------------------------------------
# 5. Build Custom CNN
# --------------------------------------------------

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        2,
        2
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        2,
        2
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        2,
        2
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# --------------------------------------------------
# 6. Compile
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# --------------------------------------------------
# 7. Callbacks
# --------------------------------------------------

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    "models/custom_cnn_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=20,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# --------------------------------------------------
# 9. Accuracy Graph
# --------------------------------------------------

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Custom CNN Accuracy")

plt.legend()

plt.savefig(
    "results/custom_cnn_accuracy.png"
)

plt.close()


# --------------------------------------------------
# 10. Loss Graph
# --------------------------------------------------

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Custom CNN Loss")

plt.legend()

plt.savefig(
    "results/custom_cnn_loss.png"
)

plt.close()


print("\nCustom CNN training completed.")
print("Best model saved at:")
print("models/custom_cnn_best.keras")
