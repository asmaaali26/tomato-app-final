import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# تحميل النموذج
@st.cache_resource
def load_model():
    file_id = '1b862FRoAlyzbz2DjpI3XeDLkeiRl_HqH'
    url = f'https://drive.google.com/uc?id={file_id}'
    model_path = 'esra.h5'
    
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    
    model = tf.keras.models.load_model(model_path)
    return model

st.title("🎯 نظام التصنيف")

# تحميل النموذج
model = load_model()

# خيارات حجم الصورة
size_option = st.selectbox(
    "اختر حجم الصورة:",
    ["96×96 (محتمل)", "64×64", "128×128", "224×224", "150×150", "80×80"]
)

# استخراج الأبعاد من الاختيار
height, width = map(int, size_option.split("×")[0].split(" ")[0].split("×"))

# تحميل الصورة
uploaded = st.file_uploader("رفع صورة", type=['jpg', 'png'])

if uploaded and model:
    image = Image.open(uploaded).convert('RGB')
    
    # تغيير الحجم
    image_resized = image.resize((width, height))
    st.image(image_resized, caption=f"الحجم: {width}×{height}")
    
    # التحويل إلى numpy
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # التنبؤ
    with st.spinner('جاري التحليل...'):
        try:
            predictions = model.predict(img_array, verbose=0)
            st.success("✅ نجح!")
            
            # عرض النتائج
            if len(predictions[0]) > 1:
                st.write("**النتائج:**")
                for i, val in enumerate(predictions[0]):
                    if val > 0.1:  # عرض القيم الكبيرة فقط
                        st.write(f"الفئة {i}: {val:.2%}")
            else:
                st.write(f"**القيمة:** {predictions[0][0]:.4f}")
                
        except Exception as e:
            st.error(f"❌ فشل مع {width}×{height}: {str(e)[:100]}")
            
            # جرب 96×96 تلقائياً (بناءً على الخطأ)
            if width != 96:
                st.info("🔄 أجرب 96×96 تلقائياً...")
                try:
                    image_resized = image.resize((96, 96))
                    img_array = np.array(image_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    predictions = model.predict(img_array, verbose=0)
                    st.success("✅ نجح مع 96×96!")
                    st.write(f"النتائج: {predictions}")
                except:
                    st.error("❌ فشل مع 96×96 أيضاً")
