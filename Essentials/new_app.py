import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import tempfile
import os

# Load the trained model
model_path = "blood_group_detector.h5"
model = load_model(model_path)

# Load the saved LabelEncoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("classes.npy", allow_pickle=True)

# Globals
img_size = 128
processed_image = None  # will store the processed image temporarily

def convert_image(image):
    """Convert uploaded image to 32-bit RGBA and resize to (103, 96)"""
    if image is None:
        return None, "No image uploaded."

    # Convert to PIL Image
    image_pil = Image.fromarray(image)
    image_32bit = image_pil.convert("RGBA")
    resized_image = image_32bit.resize((103, 96), Image.LANCZOS)

    # Save temporarily
    global processed_image
    processed_image = resized_image.convert("L").resize((img_size, img_size))  # for prediction

    return np.array(resized_image), "Image converted and resized."

def predict_blood_group():
    """Preprocess and predict from the converted image"""
    global processed_image
    if processed_image is None:
        return "Please convert the image first."

    # Convert to NumPy array and normalize
    image_np = np.array(processed_image) / 255.0
    image_np = image_np.reshape(1, img_size, img_size, 1)

    prediction = model.predict(image_np)
    predicted_class = np.argmax(prediction)
    blood_group = label_encoder.inverse_transform([predicted_class])[0]

    return f"Predicted Blood Group: {blood_group}"

# Gradio UI blocks
with gr.Blocks() as demo:
    gr.Markdown("## Blood Group Detection using CNN and Fingerprint Image")
    
    with gr.Row():
        input_image = gr.Image(type="numpy", label="Upload Fingerprint Image")
        output_image = gr.Image(label="Converted Image (103x96 RGBA)")

    convert_btn = gr.Button("Convert & Resize")
    status = gr.Textbox(label="Status")

    predict_btn = gr.Button("Predict Blood Group")
    prediction_result = gr.Textbox(label="Prediction Result")

    # Events
    convert_btn.click(fn=convert_image, inputs=input_image, outputs=[output_image, status])
    predict_btn.click(fn=predict_blood_group, inputs=[], outputs=prediction_result)

demo.launch(share=True)
