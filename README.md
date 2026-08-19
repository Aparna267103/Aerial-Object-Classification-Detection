# 🛰️ Aerial Object Classification & Detection

A Deep Learning and Computer Vision project for classifying aerial images into two categories — **Bird 🐦** and **Drone 🚁**.

The project implements a **Custom CNN** and a **Transfer Learning model** for image classification, evaluates their performance using standard classification metrics, compares both models, and deploys the best-performing model through an interactive **Streamlit** web application.

> **Domain:** Aerial Surveillance, Wildlife Monitoring, Security & Defense

---

## 📌 Project Overview

Aerial images can contain objects such as birds and drones that may look visually similar, especially from a distance. Accurate identification is important in applications such as wildlife monitoring, airport safety, restricted-airspace surveillance, and environmental research.

This project uses Deep Learning and Computer Vision techniques to distinguish between:

* 🐦 **Bird**
* 🚁 **Drone**

The project workflow includes:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Data Augmentation
   ↓
Custom CNN
   ↓
Transfer Learning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model
   ↓
Streamlit Deployment
```

The project specification also includes **YOLOv8-based object detection as an optional extension**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a binary image classification system for Bird vs Drone.
2. Preprocess and normalize aerial images.
3. Apply image augmentation to improve model generalization.
4. Develop a Custom CNN classification model.
5. Implement Transfer Learning using a pretrained deep learning model.
6. Train and validate both models.
7. Evaluate model performance using:

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix
8. Compare the Custom CNN and Transfer Learning models.
9. Select the best-performing model.
10. Deploy the classifier using Streamlit.
11. Optionally extend the project using YOLOv8 for object detection.

---

## 🛠️ Technologies Used

| Technology   | Purpose                                   |
| ------------ | ----------------------------------------- |
| Python       | Programming language                      |
| TensorFlow   | Deep Learning framework                   |
| Keras        | Neural network development                |
| NumPy        | Numerical and image-array processing      |
| Pillow       | Image processing                          |
| Scikit-learn | Model evaluation                          |
| Matplotlib   | Visualization                             |
| Seaborn      | Confusion matrix visualization            |
| Pandas       | Result analysis                           |
| Streamlit    | Web application deployment                |
| MobileNetV2  | Transfer Learning                         |
| YOLOv8       | Optional object detection                 |
| Git & GitHub | Version control and project documentation |

---

## 📂 Dataset

The project uses a binary classification dataset containing RGB `.jpg` images.

### Classification Dataset Structure

```text
dataset/
│
├── train/
│   ├── bird/
│   └── drone/
│
├── valid/
│   ├── bird/
│   └── drone/
│
└── test/
    ├── bird/
    └── drone/
```

### Dataset Distribution

| Split      | Bird | Drone |
| ---------- | ---: | ----: |
| Train      | 1414 |  1248 |
| Validation |  217 |   225 |
| Test       |  121 |    94 |

The dataset distribution above follows the project specification.

---

## 🔄 Data Preprocessing

Before training, all images are converted to a fixed input size:

```text
224 × 224 × 3
```

where:

* `224` → image height
* `224` → image width
* `3` → RGB channels

### Pixel Normalization

Original image pixel values:

```text
0 – 255
```

are normalized to:

```text
0 – 1
```

using:

```python
image = image / 255.0
```

Normalization helps provide a consistent input range for the neural network.

---

## 🔀 Data Augmentation

Data augmentation is applied to training images to introduce variations and improve model generalization.

The following transformations are used:

* Rotation
* Zoom
* Horizontal flipping

Example:

```python
ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
```

Validation and test images are normalized but are not augmented.

---

# 🧠 Model 1 — Custom CNN

The first model is a custom Convolutional Neural Network.

### Architecture

```text
Input Image
224 × 224 × 3
      ↓
Conv2D — 32 filters
      ↓
MaxPooling
      ↓
Conv2D — 64 filters
      ↓
MaxPooling
      ↓
Conv2D — 128 filters
      ↓
MaxPooling
      ↓
Flatten
      ↓
Dense — 128 neurons
      ↓
Dropout — 0.5
      ↓
Sigmoid Output
      ↓
Bird / Drone
```

### Key Components

#### Convolutional Layers

Convolution layers extract visual features such as:

* Edges
* Shapes
* Textures
* Object patterns

#### Max Pooling

Max pooling reduces spatial dimensions while retaining important features.

#### Flatten

Converts the extracted feature maps into a one-dimensional vector.

#### Dense Layer

Combines the extracted features for classification.

#### Dropout

A dropout rate of `0.5` is used to reduce overfitting during training.

#### Sigmoid Output

Since this is a binary classification problem, the final layer contains one neuron with a sigmoid activation function.

```python
Dense(1, activation="sigmoid")
```

---

# 🚀 Model 2 — Transfer Learning

The second approach uses a pretrained **MobileNetV2** model.

The pretrained convolutional base is used as a feature extractor, followed by custom classification layers.

### Architecture

```text
Input Image
224 × 224 × 3
      ↓
Pretrained MobileNetV2
      ↓
Global Average Pooling
      ↓
Dense — 128 neurons
      ↓
Dropout — 0.5
      ↓
Sigmoid Output
      ↓
Bird / Drone
```

### Why Transfer Learning?

Transfer Learning allows the project to make use of features learned by a model trained on a large image dataset.

The pretrained MobileNetV2 base is initially frozen while the classification layers are trained for the Bird vs Drone task.

---

# 🏋️ Model Training

Both models are trained using:

```text
Optimizer  → Adam
Loss       → Binary Crossentropy
Metric     → Accuracy
```

### Training Techniques

The project uses:

* Data augmentation
* Validation data
* EarlyStopping
* ModelCheckpoint

### EarlyStopping

Training can stop when validation performance stops improving, helping reduce unnecessary training and overfitting.

### ModelCheckpoint

The best-performing model based on validation accuracy is saved for later evaluation and deployment.

---

# 📊 Model Evaluation

After training, both models are evaluated using the independent test dataset.

The following metrics are calculated:

### Accuracy

Measures the overall percentage of correctly classified images.

### Precision

Measures how many predicted positive samples are actually positive.

### Recall

Measures how many actual positive samples are correctly identified.

### F1-score

Provides a balance between precision and recall.

### Confusion Matrix

Shows:

```text
                 Predicted
              Bird     Drone

Actual Bird
Actual Drone
```

This helps identify which class the model is confusing.

---

# 📈 Model Comparison

The Custom CNN and Transfer Learning models are compared based on their test performance.

The comparison includes:

| Metric          |       Custom CNN | Transfer Learning |
| --------------- | ---------------: | ----------------: |
| Accuracy        | To be calculated |  To be calculated |
| Precision       | To be calculated |  To be calculated |
| Recall          | To be calculated |  To be calculated |
| F1-score        | To be calculated |  To be calculated |
| Prediction Time | To be calculated |  To be calculated |

> The final values should be updated after running `evaluate.py` on the actual dataset.

The best-performing model is selected for Streamlit deployment based on the evaluation results.

---

# 📁 Project Structure

```text
Aerial-Object-Classification/
│
├── dataset/
│   ├── train/
│   │   ├── bird/
│   │   └── drone/
│   ├── valid/
│   │   ├── bird/
│   │   └── drone/
│   └── test/
│       ├── bird/
│       └── drone/
│
├── models/
│   ├── custom_cnn_best.keras
│   └── transfer_learning_best.keras
│
├── results/
│   ├── custom_cnn_accuracy.png
│   ├── custom_cnn_loss.png
│   ├── transfer_learning_accuracy.png
│   ├── transfer_learning_loss.png
│   ├── custom_cnn_confusion_matrix.png
│   ├── transfer_learning_confusion_matrix.png
│   └── model_comparison.csv
│
├── train.py
├── train_transfer.py
├── evaluate.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

# ▶️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd Aerial-Object-Classification
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏃 Running the Project

## Step 1 — Train Custom CNN

```bash
python train.py
```

The trained model will be saved as:

```text
models/custom_cnn_best.keras
```

Training graphs will be saved inside:

```text
results/
```

---

## Step 2 — Train Transfer Learning Model

```bash
python train_transfer.py
```

The trained model will be saved as:

```text
models/transfer_learning_best.keras
```

---

## Step 3 — Evaluate Both Models

```bash
python evaluate.py
```

The evaluation results and confusion matrices will be generated in:

```text
results/
```

The model comparison will be saved as:

```text
results/model_comparison.csv
```

---

## Step 4 — Predict a Single Image

```bash
python predict.py path/to/image.jpg
```

Example:

```bash
python predict.py dataset/test/bird/sample.jpg
```

Example output:

```text
Prediction: Bird
Confidence: 94.31%
```

---

# 🌐 Streamlit Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application allows the user to:

1. Upload an image.
2. View the uploaded image.
3. Preprocess the image.
4. Run the trained model.
5. Display the predicted class.
6. Display the prediction confidence.

Example:

```text
┌──────────────────────────────────┐
│   🛰️ Bird vs Drone Classifier   │
│                                  │
│      Upload Image                │
│                                  │
│       [ Choose File ]            │
│                                  │
│      🐦 Bird Detected            │
│      Confidence: 94.31%          │
└──────────────────────────────────┘
```

---

# 🎯 Applications

The project can be applied to several real-world scenarios.

### 🦅 Wildlife Protection

Monitoring birds near airports or wind farms to help reduce potential accidents.

### 🛡️ Security & Defense

Identifying drones operating in restricted airspace.

### ✈️ Airport Safety

Monitoring runway areas for bird activity and supporting bird-strike prevention.

### 🌱 Environmental Research

Monitoring bird populations using aerial imagery.

---

# 🔍 Optional — YOLOv8 Object Detection

The project specification also provides an optional object detection component using YOLOv8.

Unlike classification, object detection identifies both:

```text
What is the object?
+
Where is the object?
```

Example:

```text
Image
  ↓
YOLOv8
  ↓
┌─────────────────────┐
│       DRONE         │
│                     │
└─────────────────────┘
```

The object detection dataset contains YOLO-format annotations:

```text
<class_id> <x_center> <y_center> <width> <height>
```

The YOLOv8 extension can be added as a separate detection module.

---

# ⚠️ Classification vs Object Detection

| Feature                | Classification          | Object Detection     |
| ---------------------- | ----------------------- | -------------------- |
| Identifies object      | ✅                       | ✅                    |
| Locates object         | ❌                       | ✅                    |
| Output                 | Bird / Drone            | Class + Bounding Box |
| Model used             | CNN / Transfer Learning | YOLOv8               |
| Current implementation | ✅                       | Optional             |

---

# 📌 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Computer Vision
* Deep Learning
* Convolutional Neural Networks
* Binary Image Classification
* Transfer Learning
* Image Preprocessing
* Image Normalization
* Data Augmentation
* Model Training
* EarlyStopping
* ModelCheckpoint
* Performance Evaluation
* Confusion Matrix
* Precision
* Recall
* F1-score
* Model Comparison
* Streamlit Deployment
* Optional YOLOv8 Object Detection

---

# 📋 Project Deliverables

The project produces:

* Custom CNN trained model
* Transfer Learning trained model
* Evaluation results
* Confusion matrices
* Accuracy and loss plots
* Model comparison results
* Single-image prediction script
* Streamlit application
* Project documentation

---

# 🚀 Future Enhancements

Possible improvements include:

* YOLOv8 real-time object detection
* Bounding-box visualization
* Video-based detection
* Real-time webcam/stream processing
* Additional aerial object classes
* Model fine-tuning
* Improved confidence calibration
* Deployment to a cloud platform

---

# 👨‍💻 Author

Aparna V

Aerial Object Classification & Detection
Deep Learning | Computer Vision | TensorFlow | Streamlit

---

# ⭐ Acknowledgement

This project was developed as a Deep Learning and Computer Vision project focused on aerial object classification and detection.

The project specification covers aerial surveillance, wildlife monitoring, security and defense, airport bird-strike prevention, environmental research, and optional YOLOv8 object detection.

---

## 📄 License

This project is intended for educational and project demonstration purposes.
