# Aerial Object Classification & Detection

An end-to-end deep learning project for **classifying aerial images as Bird or Drone**, with an optional **YOLOv8-based object detection** component for locating and labeling objects in real-world aerial scenes.

The project combines computer vision, deep learning, transfer learning, data augmentation, model evaluation, and Streamlit deployment to address practical applications in aerial surveillance, wildlife monitoring, airport safety, and airspace security.

## 🚀 Project Overview

Accurately distinguishing birds from drones in aerial imagery is important for applications such as:

* 🦅 Wildlife protection and monitoring
* 🛡️ Security and defense surveillance
* ✈️ Airport bird-strike prevention
* 🌱 Environmental research
* 🚨 Airspace safety

The project includes two primary computer vision tasks:

1. **Image Classification** — Binary classification of aerial images into `Bird` or `Drone`.
2. **Object Detection (Optional)** — YOLOv8 detection of birds and drones with bounding boxes.

## 🧠 Key Features

* Custom CNN for image classification
* Transfer learning using pretrained architectures
* Data preprocessing and augmentation
* Model training with EarlyStopping and ModelCheckpoint
* Accuracy, Precision, Recall, and F1-score evaluation
* Confusion matrix and classification report
* Training/validation accuracy and loss visualization
* Optional YOLOv8 object detection
* Interactive Streamlit application
* Prediction confidence display
* Optional bounding-box visualization for detected objects

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras or PyTorch**
* **Computer Vision**
* **CNN**
* **Transfer Learning**
* **YOLOv8** *(optional)*
* **OpenCV**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Streamlit**

The project specification identifies Deep Learning, Computer Vision, Python, TensorFlow/Keras or PyTorch, data augmentation, YOLOv8, model evaluation, and Streamlit deployment as the main technical areas.

## 📂 Dataset

### Classification Dataset

The classification dataset contains RGB `.jpg` images divided into training, validation, and test sets.

| Split      |  Bird | Drone |
| ---------- | ----: | ----: |
| Train      | 1,414 | 1,248 |
| Validation |   217 |   225 |
| Test       |   121 |    94 |

**Task:** Binary image classification — `Bird` vs `Drone`.

### Object Detection Dataset

The optional object detection dataset contains **3,319 images** with corresponding YOLOv8-format `.txt` annotations.

Each annotation follows:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Dataset split:

| Split      | Images |
| ---------- | -----: |
| Train      |  2,662 |
| Validation |    442 |
| Test       |    215 |

> **Note:** The project specification identifies the dataset sources as `classification_dataset` and `object_detection_Dataset`.

## 🔄 Project Workflow

### 1. Dataset Exploration

* Inspect dataset structure
* Count images in each class
* Check for class imbalance
* Visualize sample images

### 2. Data Preprocessing

For classification:

* Resize images to **224 × 224**
* Normalize pixel values to `[0, 1]`
* Apply model-specific preprocessing for transfer-learning architectures
* Use ImageNet normalization where required for PyTorch pretrained models

### 3. Data Augmentation

To improve model generalization, the project uses transformations such as:

* Rotation
* Horizontal/vertical flipping
* Zoom
* Brightness adjustment
* Cropping

### 4. Model Development

#### Custom CNN

The custom CNN can include:

* Convolutional layers
* Pooling layers
* Batch normalization
* Dropout
* Dense output layer

#### Transfer Learning

Pretrained architectures such as:

* ResNet50
* MobileNet
* EfficientNetB0

can be loaded and fine-tuned for the Bird/Drone classification task.

### 5. Model Training

Models are trained using:

* EarlyStopping
* ModelCheckpoint
* Accuracy
* Precision
* Recall
* F1-score

### 6. Model Evaluation

Performance is evaluated using:

* Confusion matrix
* Classification report
* Accuracy/loss curves
* Generalization performance
* Training time

The best-performing model is selected for deployment.

### 7. Optional YOLOv8 Detection

The object detection pipeline consists of:

1. Install YOLOv8
2. Prepare the YOLOv8-format dataset
3. Create `data.yaml`
4. Train the detection model
5. Validate the model
6. Run inference on test/new images

## 🌐 Streamlit Application

The trained model can be deployed through a Streamlit web interface.

The application allows users to:

1. Upload an aerial image.
2. Run the trained classification model.
3. View the predicted class:

   * `Bird`
   * `Drone`
4. View the model's confidence score.
5. Optionally view YOLOv8 detection results with bounding boxes.

## 📁 Suggested Repository Structure

```text
Aerial-Object-Classification-Detection/
│
├── data/
│   ├── classification_dataset/
│   │   ├── train/
│   │   │   ├── bird/
│   │   │   └── drone/
│   │   ├── valid/
│   │   │   ├── bird/
│   │   │   └── drone/
│   │   └── test/
│   │       ├── bird/
│   │       └── drone/
│   │
│   └── object_detection_Dataset/
│       ├── images/
│       ├── labels/
│       └── data.yaml
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── cnn_classification.ipynb
│   ├── transfer_learning.ipynb
│   └── yolov8_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── custom_cnn/
│   ├── transfer_learning/
│   └── yolov8/
│
├── app/
│   └── streamlit_app.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── classification_report.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

*This is a suggested GitHub organization; the source project specification requires trained models, a Streamlit application, scripts/notebooks, evaluation/model comparison documentation, and a well-structured GitHub repository, but does not prescribe this exact folder structure.*

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Aerial-Object-Classification-Detection.git
cd Aerial-Object-Classification-Detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Streamlit App

After training and saving the best-performing model:

```bash
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL displayed in the terminal.

Upload an aerial image to receive a **Bird/Drone prediction and confidence score**. If YOLOv8 detection has been implemented, the application can additionally display detected objects with bounding boxes.

## 📊 Model Comparison

The project compares the developed classification models using:

| Metric         | Custom CNN | Transfer Learning |
| -------------- | ---------- | ----------------- |
| Accuracy       | —          | —                 |
| Precision      | —          | —                 |
| Recall         | —          | —                 |
| F1-Score       | —          | —                 |
| Training Time  | —          | —                 |
| Generalization | —          | —                 |

Replace the placeholders with the **actual experimental results** after training. The project specification requires comparison based on accuracy, training time, and generalization performance.

## 🎯 Applications

### Wildlife Protection

Identify birds around wind farms and airports to help reduce potential accidents.

### Security & Defense

Detect drones operating in restricted airspace and support timely surveillance alerts.

### Airport Safety

Monitor runway and surrounding areas for bird activity to support bird-strike prevention.

### Environmental Research

Use aerial imagery to monitor bird populations while reducing bird/drone misclassification.

## 📌 Project Deliverables

* [ ] Trained Custom CNN model
* [ ] Trained Transfer Learning model
* [ ] YOLOv8 model *(optional)*
* [ ] Streamlit classification/detection application
* [ ] Preprocessing and training scripts/notebooks
* [ ] Evaluation results
* [ ] Model comparison report
* [ ] Well-structured and commented code
* [ ] GitHub documentation

These deliverables follow the project specification.

## 🧪 Future Improvements

Potential extensions include:

* Real-time video detection
* Additional aerial-object classes
* Improved model optimization
* Edge-device deployment
* Real-time surveillance alerts
* Larger and more diverse aerial datasets
* Further YOLOv8 optimization

## 🏷️ Technical Tags

`Computer Vision` · `Deep Learning` · `Image Classification` · `Object Detection` · `CNN` · `YOLOv8` · `Transfer Learning` · `Data Augmentation` · `Model Evaluation` · `Streamlit` · `Aerial Surveillance AI`

## 👨‍💻 Author

Aparna V

---

## 📄 Project Reference

**Project:** Aerial Object Classification & Detection

**Domain:** Aerial Surveillance, Wildlife Monitoring, Security & Defense

**Primary Tasks:** Bird/Drone Image Classification + Optional Object Detection

**Deployment:** Streamlit
