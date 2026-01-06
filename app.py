import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# =====================================================
# Custom Styling
# =====================================================
st.markdown("""
<style>
.main {
    background-color: #f7fafc;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #22543d;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #4a5568;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.result {
    font-size: 24px;
    font-weight: bold;
    color: #2f855a;
}
.confidence {
    font-size: 18px;
    color: #276749;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Load Model (SAFE & FINAL)
# =====================================================
@st.cache_resource
def load_cnn_model():
    model_path = "esra_fixed.h5"

    if not os.path.exists(model_path):
        st.error("❌ Model file (esra.h5) not found in project folder.")
        st.stop()

    model = tf.keras.models.load_model(
        model_path,
        compile=False   # ⭐ حل نهائي لمشكلة InputLayer
    )
    return model


model = load_cnn_model()

# =====================================================
# Header
# =====================================================
st.markdown('<div class="title">🍅 Tomato Disease Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">CNN-based classification of tomato leaf diseases</div>',
    unsafe_allow_html=True
)

# =====================================================
# Class Labels (عدّليها لو مختلفة)
# =====================================================
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
    "Healthy"
]

# =====================================================
# Image Preprocessing
# =====================================================
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =====================================================
# Upload Section
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# Prediction Section
# =====================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            img = preprocess_image(image)
            preds = model.predict(img)

            idx = np.argmax(preds)
            confidence = np.max(preds) * 100

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="result">🧪 Disease: {class_names[idx]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="confidence">📊 Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.markdown(
    "🎓 **Graduation Project – Tomato Disease Detection using CNN**  \n"
    "Developed using TensorFlow & Streamlit"
)
st.caption(f"TensorFlow version: {tf.__version__}")
