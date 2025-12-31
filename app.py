import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tempfile

st.set_page_config(page_title="Tomato Disease Classifier", page_icon="🍅", layout="wide")

# ===== إعدادات التطبيق =====
st.sidebar.title("⚙️ الإعدادات")
st.sidebar.info("تطبيق تصنيف أمراض الطماطم باستخدام الذكاء الاصطناعي")

MODEL_URL = "https://drive.google.com/file/d/1b862FRoAlyzbz2DjpI3XeDLkeiRl_HqH/view?usp=sharing"  # ضع ID ملفك هنا
IMAGE_SIZE = (256, 256)

CLASS_NAMES = [
    'Bacterial_spot', 
    'Early_blight', 
    'Late_blight', 
    'Leaf_Mold', 
    'Septoria_leaf_spot', 
    'Spider_mites Two-spotted_spider_mite', 
    'Target_Spot', 
    'Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_healthy', 
    'Tomato_mosaic_virus'
]

# ===== تحميل المودل من Google Drive =====
def download_file_from_google_drive(file_id, destination):
    """تحميل ملف من Google Drive بدون استخدام gdown"""
    URL = "https://drive.google.com/uc?export=download"
    
    with requests.Session() as session:
        response = session.get(URL, params={'id': file_id}, stream=True)
        
        # معالجة ملفات Google Drive الكبيرة
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(URL, params=params, stream=True)
                break
        
        # الحصول على حجم الملف
        total_size = int(response.headers.get('content-length', 0))
        
        # شريط التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # حفظ الملف
        downloaded = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # تحديث شريط التقدم
                    if total_size:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"جاري التحميل: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()

@st.cache_resource
def load_model():
    """تحميل المودل مع معالجة الأخطاء"""
    model_path = "model.h5"
    
    # إذا لم يكن المودل موجوداً، حمله
    if not os.path.exists(model_path):
        st.info("📥 جاري تحميل المودل لأول مرة...")
        try:
            file_id = MODEL_URL.split('id=')[1] if 'id=' in MODEL_URL else MODEL_URL.split('/')[-2]
            download_file_from_google_drive(file_id, model_path)
            st.success("✅ تم تحميل المودل بنجاح!")
        except Exception as e:
            st.error(f"❌ فشل تحميل المودل: {e}")
            return None
    
    try:
        import tensorflow as tf
        # محاولة تحميل المودل
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success("✅ المودل جاهز للاستخدام")
        return model
    except Exception as e:
        st.error(f"❌ خطأ في تحميل المودل: {str(e)[:200]}...")
        st.info("حاول استخدام رابط Google Drive صحيح")
        return None

# ===== الواجهة الرئيسية =====
st.title("🍅 Tomato Plant Disease Classifier")
st.markdown("---")

# تحميل المودل
model = load_model()

if model is None:
    st.error("تعذر تحميل المودل. تأكد من رابط Google Drive.")
    st.info("""
    **خطوات حل المشكلة:**
    1. تأكد أن ملف `last.h5` موجود على Google Drive
    2. غير إعدادات المشاركة إلى "أي شخص لديه الرابط"
    3. انسخ ID الملف من الرابط
    4. أضعه في المتغير `MODEL_URL`
    """)
    st.stop()

# ===== قسم رفع الصور =====
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 رفع الصورة")
    uploaded_file = st.file_uploader(
        "اختر صورة ورقة الطماطم",
        type=["jpg", "jpeg", "png"],
        help="ارفع صورة واضحة لورقة الطماطم"
    )
    
    if uploaded_file is not None:
        # عرض الصورة
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        
        # معالجة الصورة
        img_array = np.array(image.resize(IMAGE_SIZE)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

with col2:
    if uploaded_file is not None:
        st.subheader("🔍 النتائج")
        
        # زر التنبؤ
        if st.button("🚀 بدأ التحليل", type="primary", use_container_width=True):
            with st.spinner("جاري تحليل الصورة..."):
                try:
                    # التنبؤ
                    predictions = model.predict(img_array, verbose=0)[0]
                    
                    # العثور على أعلى نتيجة
                    predicted_idx = np.argmax(predictions)
                    confidence = predictions[predicted_idx] * 100
                    disease_name = CLASS_NAMES[predicted_idx]
                    
                    # عرض النتيجة
                    st.markdown(f"### 📊 النتيجة:")
                    
                    if "healthy" in disease_name.lower():
                        st.success(f"✅ **النبات سليم**")
                        st.balloons()
                    else:
                        st.error(f"⚠️ **المرض المتوقع:** {disease_name}")
                    
                    st.info(f"**مستوى الثقة:** {confidence:.2f}%")
                    
                    # عرض جميع النتائج
                    st.markdown("---")
                    st.subheader("📈 جميع الاحتمالات")
                    
                    results = []
                    for i, (name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                        results.append({
                            "المرض": name,
                            "النسبة المئوية": f"{prob*100:.2f}%",
                            "القيمة": prob*100
                        })
                    
                    results_df = pd.DataFrame(results).sort_values("القيمة", ascending=False)
                    st.dataframe(results_df[["المرض", "النسبة المئوية"]], use_container_width=True)
                    
                    # رسم بياني
                    st.bar_chart(results_df.set_index("المرض")["القيمة"])
                    
                except Exception as e:
                    st.error(f"❌ حدث خطأ أثناء التحليل: {e}")

# ===== قسم المعلومات =====
st.markdown("---")
with st.expander("ℹ️ معلومات عن الأمراض"):
    st.write("""
    **قائمة الأمراض التي يمكن الكشف عنها:**
    
    1. **Bacterial Spot** - بقعة بكتيرية
    2. **Early Blight** - اللفحة المبكرة  
    3. **Late Blight** - اللفحة المتأخرة
    4. **Leaf Mold** - عفن الأوراق
    5. **Septoria Leaf Spot** - بقعة سبتوريا
    6. **Spider Mites** - العناكب
    7. **Target Spot** - البقعة الهدفية
    8. **Yellow Leaf Curl Virus** - فيروس تجعد الأوراق الأصفر
    9. **Mosaic Virus** - فيروس الموزاييك
    10. **Healthy** - سليم
    """)

# ===== قسم المساعدة =====
with st.sidebar.expander("🆘 المساعدة التقنية"):
    st.write("""
    **إذا واجهت مشاكل:**
    
    1. تأكد من جودة الصورة
    2. تأكد من أن الصورة لورقة طماطم
    3. إذا ظهر خطأ، أعد تحميل الصفحة
    4. تأكد من اتصال الإنترنت
    """)

st.sidebar.markdown("---")
st.sidebar.caption("تم التطوير باستخدام TensorFlow و Streamlit")
