import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# =====================================
# Custom CSS for Better UI
# =====================================
st.markdown("""
<style>
.main {
    background-color: #f9fafb;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2c7a7b;
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
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.result {
    font-size: 22px;
    font-weight: bold;
    color: #22543d;
}
.confidence {
    font-size: 18px;
    color: #2f855a;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# Load CNN Model (SAFE WAY)
# =====================================
@st.cache_resource
def load_cnn_model():
    model_path = "esra.h5"   # ⭐ اسم الموديل الصحيح

    if not os.path.exists(model_path):
        st.error("❌ Model file (esra.h5) not found. Please upload it to the project folder.")
        st.stop()

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False   # ⭐ حل مشكلة InputLayer / batch_shape
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


model = load_cnn_model()

# =====================================
# Header
# =====================================
st.markdown('<div class="title">🍅 Tomato Disease Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">CNN-based classification of tomato leaf diseases</div>',
    unsafe_allow_html=True
)

# =====================================
# Class Names (عدليهم لو مختلفين)
# =====================================
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

# =====================================
# Image Preprocessing
# =====================================
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =====================================
# Upload Section
# =====================================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# Prediction Section
# =====================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)

            class_index = np.argmax(predictions)
            confidence = np.max(predictions) * 100

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result">🧪 Disease: {class_names[class_index]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="confidence">📊 Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# Footer
# =====================================
st.markdown("---")
st.markdown(
    "🎓 **Graduation Project – Tomato Disease Detection using CNN**  \n"
    "Developed with TensorFlow & Streamlit"
)

st.caption(f"TensorFlow version: {tf.__version__}")
