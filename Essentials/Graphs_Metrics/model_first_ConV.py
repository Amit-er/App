import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model

# ---------- Step 1: Load and preprocess dataset ----------
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

# Path and size setup
data_path = "dataset"  # Your dataset folder
img_size = 128

# Load and preprocess
data, labels = load_dataset(data_path, img_size)
data = data / 255.0
data = data.reshape(-1, img_size, img_size, 1)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels_categorical, test_size=0.2, random_state=42)

# ---------- Step 2: Load model ----------
model = tf.keras.models.load_model("blood_group_detector.h5")

# ---------- Step 3: Feature map visualization ----------
# Pick one image from test set
sample_img = X_test[0].reshape(1, img_size, img_size, 1)

# Get Conv2D layer outputs
conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
layer_outputs = [layer.output for layer in conv_layers]
layer_names = [layer.name for layer in conv_layers]

# Create a model that returns outputs of conv layers
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Predict activations
activations = activation_model.predict(sample_img)

# Plot feature maps for each Conv layer
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    cols = 8
    rows = int(np.ceil(n_features / cols))

    plt.figure(figsize=(15, rows * 2))
    for i in range(n_features):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(layer_activation[0, :, :, i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f"F{i}", fontsize=6)

    plt.suptitle(f"Feature Maps - Layer: {layer_name}", fontsize=14)
    plt.tight_layout()
    plt.show()
