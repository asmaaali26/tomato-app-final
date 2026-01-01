import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# تأكد من وجود المودل
if not os.path.exists('last.h5'):
    st.error("Model file 'last.h5' not found!")
    st.stop()

# تحميل المودل
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('esra.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

if model is None:
    st.error("Could not load AI model. Using simulation mode.")
    # استخدم الوضع الوهمي كاحتياطي
else:
    st.success(f"✅ AI Model Loaded Successfully!")
    st.info(f"Model input shape: {model.input_shape}")


