import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

test_dir = "dataset/test"

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

model = tf.keras.models.load_model("models/bird_drone_model.h5")

preds = model.predict(test_data)
y_pred = np.where(preds > 0.5, 1, 0)

print(confusion_matrix(test_data.classes, y_pred))
print(classification_report(test_data.classes, y_pred))