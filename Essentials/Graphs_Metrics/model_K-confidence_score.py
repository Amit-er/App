import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model("blood_group_detector.h5")

# Load dataset again (same way as in your code)
def load_dataset(dataset_path, img_size):
    data = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    data.append(img)
                    labels.append(label)
    
    return np.array(data), np.array(labels)

data_path = "dataset"
img_size = 128
data, labels = load_dataset(data_path, img_size)
data = data / 255.0
data = data.reshape(-1, img_size, img_size, 1)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(data, labels_categorical, test_size=0.2, random_state=42)
# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

top_preds = np.max(y_pred, axis=1)
plt.hist(top_preds, bins=20, color='lightgreen', edgecolor='black')
plt.title("Prediction Confidence Distribution")
plt.xlabel("Top-1 Confidence Score")
plt.ylabel("Number of Predictions")
plt.grid(True)
plt.show()