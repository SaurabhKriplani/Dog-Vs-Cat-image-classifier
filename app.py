import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

if not os.path.exists("dog_cat_classifier.h5"):
    url = "https://drive.google.com/uc?export=download&id=1KIwXQPv9G4lNPmtd2y2ktSqWjyBxdids"
    gdown.download(url, "dog_cat_classifier.h5", quiet=False)
# Load model

model = tf.keras.models.load_model("dog_cat_classifier.h5")

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
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img, verbose=0)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.success("🐶 Prediction: Dog")
        st.write(f"Confidence: {confidence * 100:.2f}%")
    else:
        st.success("🐱 Prediction: Cat")
        st.write(f"Confidence: {(1 - confidence) * 100:.2f}%")