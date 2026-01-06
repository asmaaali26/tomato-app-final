import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =======================
# Page config
# =======================
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# =======================
# UI Styling
# =======================
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .title {
        text-align: center;
        color: #2e7d32;
        font-size: 40px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🍅 Tomato Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CNN-based Image Classification</div>', unsafe_allow_html=True)
st.write("")

# =======================
# Load model (SAFE MODE)
# =======================
@st.cache_resource
def load_model_safe():
    model = tf.keras.models.load_model(
        "esra.h5",
        compile=False   # 🔴 ده أهم سطر
    )
    return model


try:
    model = load_model_safe()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Model failed to load")
    st.exception(e)
    st.stop()

# =======================
# Class labels (عدليها حسب موديلك)
# =======================
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy"
]

# =======================
# Image upload
# =======================
uploaded_file = st.file_uploader(
    "📤 Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    st.write(f"**Disease:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
