import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tempfile
import gdown
import os

# عنوان التطبيق
st.set_page_config(page_title="نظام التصنيف الذكي", page_icon="🤖", layout="wide")

# تحميل النموذج (مع التخزين المؤقت)
@st.cache_resource
def load_model():
    # تحميل النموذج من Google Drive
    file_id = '1b862FRoAlyzbz2DjpI3XeDLkeiRl_HqH'
    url = f'https://drive.google.com/uc?id={file_id}'
    model_path = 'esra.h5'
    
    # تحميل الملف إذا لم يكن موجوداً
    if not os.path.exists(model_path):
        with st.spinner('جاري تحميل النموذج من السحابة...'):
            gdown.download(url, model_path, quiet=False)
    
    # تحميل النموذج
    model = tf.keras.models.load_model(model_path)
    
    # تجميع النموذج إذا لزم الأمر
    try:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    except:
        pass
    
    return model

# معالجة الصورة
def preprocess_image(image, target_size=(224, 224)):
    """
    تحويل الصورة إلى الشكل المناسب للنموذج
    """
    # تحويل إلى RGB إذا كان ARGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # تغيير الحجم
    image = image.resize(target_size)
    
    # تحويل إلى numpy array
    img_array = np.array(image)
    
    # تطبيع القيم (0-1)
    img_array = img_array / 255.0
    
    # إضافة بُعد الدُفعة (batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# الواجهة الرئيسية
def main():
    st.title("🎨 نظام التصنيف الذكي بالذكاء الاصطناعي")
    st.markdown("---")
    
    # تحميل النموذج
    try:
        model = load_model()
        st.success("✅ تم تحميل النموذج بنجاح!")
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return
    
    # عرض معلومات النموذج
    with st.expander("📊 معلومات النموذج"):
        st.write(f"**اسم النموذج:** {model.name}")
        st.write(f"**عدد الطبقات:** {len(model.layers)}")
        st.write(f"**شكل الإدخال:** {model.input_shape}")
        st.write(f"**شكل الإخراج:** {model.output_shape}")
    
    # الشريط الجانبي
    st.sidebar.title("⚙️ الإعدادات")
    
    # خيارات التحميل
    upload_option = st.sidebar.radio(
        "اختر طريقة تحميل الصورة:",
        ["📤 رفع صورة", "📷 استخدام الكاميرا"]
    )
    
    # القائمة المنسدلة للفئات (عدلها حسب فئات نموذجك)
    class_names = st.sidebar.multiselect(
        "اختر الفئات المتوقعة:",
        ["قطة", "كلب", "سيارة", "شجرة", "منزل", "وجه", "كتاب", "زهرة", "طائرة", "قارب"],
        default=["قطة", "كلب", "زهرة"]
    )
    
    # درجة الثقة
    confidence_threshold = st.sidebar.slider(
        "📊 حد الثقة:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # منطقة العرض الرئيسية
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📷 الإدخال")
        
        # تحميل الصورة
        image = None
        
        if upload_option == "📤 رفع صورة":
            uploaded_file = st.file_uploader(
                "اختر صورة...", 
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="الصور المدعومة: JPG, PNG, BMP"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        
        else:  # استخدام الكاميرا
            camera_image = st.camera_input("التقط صورة باستخدام الكاميرا")
            if camera_image is not None:
                image = Image.open(camera_image)
    
    with col2:
        st.header("📈 النتائج")
        
        if image is not None:
            with st.spinner('🔄 جاري تحليل الصورة...'):
                # معالجة الصورة
                processed_image = preprocess_image(image)
                
                # التنبؤ
                predictions = model.predict(processed_image, verbose=0)
                
                # استخراج النتائج
                if len(predictions[0]) > 1:  # تصنيف متعدد الفئات
                    # الحصول على أعلى 3 تنبؤات
                    top_indices = np.argsort(predictions[0])[-3:][::-1]
                    top_values = predictions[0][top_indices]
                    
                    # عرض النتائج
                    st.success("✅ تم تحليل الصورة بنجاح!")
                    
                    # رسم تقدمي للنتائج
                    for i, (idx, conf) in enumerate(zip(top_indices, top_values)):
                        if conf > confidence_threshold:
                            # شريط التقدم
                            st.progress(float(conf))
                            
                            # اسم الفئة (استخدم الفهرس إذا لم تكن الأسماء محددة)
                            class_name = f"الفئة {idx}" if len(class_names) <= idx else class_names[idx]
                            
                            # عرض النتيجة
                            st.metric(
                                label=f"**{class_name}**",
                                value=f"{conf*100:.2f}%",
                                delta="عالية" if conf > 0.8 else "متوسطة" if conf > 0.5 else "منخفضة"
                            )
                    
                    # عرض تنبؤ مفصل
                    with st.expander("📋 تفاصيل التنبؤات الكاملة"):
                        for idx, conf in enumerate(predictions[0]):
                            if conf > 0.01:  # عرض القيم الأكبر من 1%
                                class_name = f"الفئة {idx}" if len(class_names) <= idx else class_names[idx]
                                st.write(f"{class_name}: {conf*100:.2f}%")
                
                else:  # تصنيف ثنائي
                    confidence = float(predictions[0][0])
                    st.success(f"**الثقة:** {confidence*100:.2f}%")
                    
                    if confidence > confidence_threshold:
                        st.balloons()
                        st.success("✅ النتيجة إيجابية")
                    else:
                        st.warning("⚠️ النتيجة سلبية")
        
        else:
            st.info("⬅️ يرجى تحميل صورة لرؤية النتائج")
    
    # قسم التحميلات
    st.markdown("---")
    st.header("📥 تحميل النموذج يدوياً")
    
    # خيار لتحميل نموذج مختلف
    st.write("إذا كنت تريد استخدام نموذج مختلف:")
    new_model_url = st.text_input(
        "رابط Google Drive للنموذج:",
        value="https://drive.google.com/uc?id=1b862FRoAlyzbz2DjpI3XeDLkeiRl_HqH"
    )
    
    if st.button("🔄 تحديث النموذج"):
        with st.spinner('جاري تحديث النموذج...'):
            try:
                # استخراج file_id من الرابط
                if "id=" in new_model_url:
                    file_id = new_model_url.split("id=")[1]
                else:
                    # أو استخراج من رابط المشاركة
                    file_id = new_model_url.split("/d/")[1].split("/")[0]
                
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, 'esra.h5', quiet=False)
                
                # مسح ذاكرة التخزين المؤقت لإعادة التحميل
                st.cache_resource.clear()
                
                st.success("✅ تم تحديث النموذج بنجاح! يرجى إعادة تحميل الصفحة.")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ خطأ في تحديث النموذج: {e}")

# تشغيل التطبيق
if __name__ == "__main__":
    main()
