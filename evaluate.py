import os
import time

import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

TEST_DIR = "dataset/test"

os.makedirs(
    "results",
    exist_ok=True
)


# --------------------------------------------------
# 2. Test Data for Custom CNN
# --------------------------------------------------

custom_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


custom_test_data = custom_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)


# --------------------------------------------------
# 3. Test Data for MobileNetV2
# --------------------------------------------------

transfer_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


transfer_test_data = transfer_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)


print(
    "Class Mapping:",
    custom_test_data.class_indices
)


# --------------------------------------------------
# 4. Models
# --------------------------------------------------

models_to_evaluate = {

    "Custom CNN": (
        "models/custom_cnn_best.keras",
        custom_test_data
    ),

    "Transfer Learning": (
        "models/transfer_learning_best.keras",
        transfer_test_data
    )
}


results = []


# --------------------------------------------------
# 5. Evaluate Loop
# --------------------------------------------------

for model_name, (
    model_path,
    test_data
) in models_to_evaluate.items():

    print("\n" + "=" * 60)

    print(model_name)

    print("=" * 60)


    # Load model
    model = tf.keras.models.load_model(
        model_path
    )


    # Reset test generator
    test_data.reset()


    # Start timer
    start_time = time.time()


    # Predict
    predictions = model.predict(
        test_data,
        verbose=0
    )


    # Prediction time
    prediction_time = (
        time.time() - start_time
    )


    # Convert probability to class
    y_pred = (
        predictions.ravel() > 0.5
    ).astype(int)


    # Actual labels
    y_true = test_data.classes


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    print(
        "\nAccuracy:",
        accuracy
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "F1 Score:",
        f1
    )


    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    print(
        "\nClassification Report:\n"
    )


    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Bird",
                "Drone"
            ],
            zero_division=0
        )
    )


    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )


    print(
        "\nConfusion Matrix:"
    )

    print(cm)


    # --------------------------------------------------
    # Plot Confusion Matrix
    # --------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=[
            "Bird",
            "Drone"
        ],
        yticklabels=[
            "Bird",
            "Drone"
        ]
    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    plt.title(
        model_name +
        " Confusion Matrix"
    )


    filename = (
        model_name
        .lower()
        .replace(" ", "_")
    )


    plt.savefig(
        f"results/{filename}_confusion_matrix.png"
    )


    plt.close()


    # --------------------------------------------------
    # Store Results Metrics
    # --------------------------------------------------

    results.append({

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1,

        "Prediction Time (seconds)":
            prediction_time
    })


# --------------------------------------------------
# 6. Comparison
# --------------------------------------------------

comparison = pd.DataFrame(
    results
)


print("\n" + "=" * 60)

print(
    "MODEL COMPARISON"
)

print("=" * 60)


print(
    comparison
)


# --------------------------------------------------
# 7. Save CSV
# --------------------------------------------------

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)


print(
    "\nComparison saved to:"
)

print(
    "results/model_comparison.csv"
)
