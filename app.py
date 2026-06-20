import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf


# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="dog_cat_classifier.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐶"
)

st.title("🐶 Dog vs 🐱 Cat Classifier")
st.write("Upload an image and let the AI predict whether it's a dog or a cat.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    img = image.resize((256, 256))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing image..."):
        interpreter.set_tensor(
            input_details[0]['index'],
            img
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]['index']
        )

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.success("🐶 Prediction: Dog")
        st.write(f"Confidence: {confidence * 100:.2f}%")
    else:
        st.success("🐱 Prediction: Cat")
        st.write(f"Confidence: {(1 - confidence) * 100:.2f}%")