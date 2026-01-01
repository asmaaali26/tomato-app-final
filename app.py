import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gdown
import os
import requests
from io import BytesIO

# عنوان التطبيق
st.set_page_config(
    page_title="نظام التصنيف الذكي",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-right: 4px solid #10B981;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-right: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# تحميل النموذج (مع التخزين المؤقت)
@st.cache_resource
def load_model():
    try:
        # تحميل النموذج من Google Drive
        file_id = '1b862FRoAlyzbz2DjpI3XeDLkeiRl_HqH'
        url = f'https://drive.google.com/uc?id={file_id}'
        model_path = 'esra.h5'
        
        # تحميل الملف إذا لم يكن موجوداً
        if not os.path.exists(model_path):
            with st.spinner('🔄 جاري تحميل النموذج من السحابة...'):
                gdown.download(url, model_path, quiet=False)
        
        # تحميل النموذج
        model = tf.keras.models.load_model(model_path)
        
        # تجميع النموذج
        try:
            model.compile(optimizer='adam')
        except:
            pass
        
        return model
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None

# معالجة الصورة باستخدام Pillow فقط
def preprocess_image(image, target_size=(224, 224)):
    """
    تحويل الصورة إلى الشكل المناسب للنموذج باستخدام Pillow فقط
    """
    # تحويل إلى RGB إذا كان في صيغة أخرى
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # تغيير الحجم
    image = image.resize(target_size)
    
    # تحويل إلى numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # تطبيع القيم (0-1)
    img_array = img_array / 255.0
    
    # إضافة بُعد الدُفعة (batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# رسم مربع على الصورة باستخدام Pillow
def draw_box_on_image(image, box, label="", color="green", thickness=3):
    """
    رسم مربع على الصورة باستخدام Pillow فقط
    """
    draw = ImageDraw.Draw(image)
    
    # تحويل اللون من سلسلة إلى RGB
    color_map = {
        "green": (0, 255, 0),
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0)
    }
    rgb_color = color_map.get(color, (0, 255, 0))
    
    # رسم المربع
    draw.rectangle(box, outline=rgb_color, width=thickness)
    
    # إضافة تسمية إذا كانت موجودة
    if label:
        try:
            # محاولة تحميل خط عربي
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # استخدام الخط الافتراضي
            font = ImageFont.load_default()
        
        # خلفية للتسمية
        text_bbox = draw.textbbox((box[0], box[1] - 25), label, font=font)
        draw.rectangle(text_bbox, fill=rgb_color)
        
        # النص
        draw.text((box[0], box[1] - 25), label, fill="white", font=font)
    
    return image

# الواجهة الرئيسية
def main():
    st.markdown('<h1 class="main-header">🤖 نظام التصنيف الذكي بالذكاء الاصطناعي</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">قم بتحميل صورة لتحليلها باستخدام نموذج الذكاء الاصطناعي المتقدم</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # تحميل النموذج
    with st.spinner('⏳ جاري تحميل النموذج...'):
        model = load_model()
    
    if model is None:
        st.error("""
        ### ❌ تعذر تحميل النموذج
        
        **الحلول المقترحة:**
        1. تأكد من وجود اتصال بالإنترنت
        2. تحقق من رابط النموذج
        3. حاول تحديث الصفحة
        """)
        return
    
    st.markdown('<div class="success-box">✅ تم تحميل النموذج بنجاح!</div>', unsafe_allow_html=True)
    
    # عرض معلومات النموذج
    with st.expander("📊 معلومات النموذج", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("اسم النموذج", "Esra Model")
            st.metric("عدد الطبقات", len(model.layers))
        with col2:
            st.metric("شكل الإدخال", str(model.input_shape))
            st.metric("شكل الإخراج", str(model.output_shape))
    
    # الشريط الجانبي
    with st.sidebar:
        st.title("⚙️ الإعدادات")
        
        # خيارات التحميل
        st.markdown("### 📤 طريقة التحميل")
        upload_option = st.radio(
            "اختر طريقة تحميل الصورة:",
            ["رفع صورة من الجهاز", "إدخال رابط الصورة", "استخدام صورة مثال"],
            index=0,
            label_visibility="collapsed"
        )
        
        # إعدادات التنبؤ
        st.markdown("### 📈 إعدادات التنبؤ")
        
        # درجة الثقة
        confidence_threshold = st.slider(
            "حد الثقة المطلوب",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="كلما زادت قيمة الثقة، زادت دقة التنبؤات المعروضة"
        )
        
        # عدد النتائج المراد عرضها
        num_results = st.slider(
            "عدد النتائج المعروضة",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        # خيارات إضافية
        st.markdown("### 🔧 خيارات إضافية")
        show_details = st.checkbox("عرض تفاصيل التنبؤات الكاملة", value=True)
        save_results = st.checkbox("حفظ النتائج", value=False)
    
    # منطقة العرض الرئيسية
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📷 الإدخال")
        
        # تحميل الصورة بناءً على الخيار المحدد
        image = None
        image_source = None
        
        if upload_option == "رفع صورة من الجهاز":
            uploaded_file = st.file_uploader(
                "اسحب وأفلت الصورة هنا أو انقر للاختيار",
                type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
                help="الصور المدعومة: JPG, PNG, BMP, GIF"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_source = uploaded_file.name
        
        elif upload_option == "إدخال رابط الصورة":
            image_url = st.text_input(
                "أدخل رابط الصورة:",
                placeholder="https://example.com/image.jpg",
                help="يجب أن يكون الرابط مباشراً للصورة"
            )
            
            if image_url:
                try:
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        image_source = image_url
                    else:
                        st.error(f"❌ فشل في تحميل الصورة: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ خطأ في تحميل الصورة: {e}")
        
        else:  # استخدام صورة مثال
            example_images = {
                "قطة": "https://images.unsplash.com/photo-1514888286974-6d03bdeacba8?w=400",
                "زهرة": "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w-400",
                "سيارة": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=400"
            }
            
            selected_example = st.selectbox(
                "اختر صورة مثال:",
                list(example_images.keys())
            )
            
            if selected_example:
                try:
                    response = requests.get(example_images[selected_example], timeout=10)
                    image = Image.open(BytesIO(response.content))
                    image_source = f"مثال: {selected_example}"
                except:
                    # استخدام صورة افتراضية إذا فشل التحميل
                    st.info("⚙️ جاري تحضير صورة المثال...")
                    # إنشاء صورة افتراضية
                    image = Image.new('RGB', (224, 224), color='lightblue')
                    draw = ImageDraw.Draw(image)
                    draw.text((80, 100), f"صورة {selected_example}", fill="black")
                    image_source = f"مثال: {selected_example}"
        
        # عرض الصورة
        if image is not None:
            st.image(image, caption=f"📷 {image_source}", use_column_width=True)
            
            # عرض معلومات الصورة
            with st.expander("📄 معلومات الصورة"):
                st.write(f"**الحجم:** {image.size[0]} × {image.size[1]} بكسل")
                st.write(f"**النمط:** {image.mode}")
                st.write(f"**التنسيق:** {image.format if hasattr(image, 'format') else 'غير معروف'}")
    
    with col2:
        st.header("📈 النتائج")
        
        if image is not None:
            with st.spinner('🔄 جاري تحليل الصورة...'):
                try:
                    # معالجة الصورة
                    processed_image = preprocess_image(image)
                    
                    # التنبؤ
                    predictions = model.predict(processed_image, verbose=0)
                    
                    # استخراج النتائج
                    predictions_array = predictions[0]
                    
                    # الحصول على أفضل النتائج
                    top_indices = np.argsort(predictions_array)[-num_results:][::-1]
                    top_values = predictions_array[top_indices]
                    
                    # عرض النتائج
                    st.markdown('<div class="info-box">✅ تم تحليل الصورة بنجاح!</div>', unsafe_allow_html=True)
                    
                    # عرض أفضل النتائج مع أشرطة التقدم
                    st.subheader("🎯 أفضل التنبؤات:")
                    
                    for i, (idx, conf) in enumerate(zip(top_indices, top_values)):
                        if conf > confidence_threshold:
                            # شريط التقدم
                            progress_bar = st.progress(float(conf))
                            
                            # اسم الفئة
                            class_name = f"الفئة {idx + 1}"
                            
                            # عرض النتيجة
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"**{class_name}**")
                            with col_b:
                                st.write(f"**{conf*100:.2f}%**")
                            
                            # شريط أفقي ملون
                            color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                            st.write(f"{color} {'█' * int(conf*20)}")
                            
                            st.write("---")
                    
                    # عرض تفاصيل التنبؤات الكاملة
                    if show_details:
                        with st.expander("📋 تفاصيل التنبؤات الكاملة", expanded=False):
                            for idx, conf in enumerate(predictions_array):
                                if conf > 0.01:  # عرض القيم الأكبر من 1%
                                    st.write(f"الفئة {idx + 1}: {conf*100:.2f}%")
                    
                    # خيارات إضافية للنتائج
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        if st.button("🎉 عرض البالونات"):
                            st.balloons()
                            st.success("🎊 تم التحليل بنجاح!")
                    
                    with col4:
                        if st.button("🔄 إعادة التحليل"):
                            st.rerun()
                    
                    # حفظ النتائج
                    if save_results:
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"results_{timestamp}.txt"
                        
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(f"نتائج تحليل الصورة - {timestamp}\n")
                            f.write(f"الصورة: {image_source}\n")
                            f.write("-" * 50 + "\n")
                            for idx, conf in zip(top_indices, top_values):
                                f.write(f"الفئة {idx + 1}: {conf*100:.2f}%\n")
                        
                        st.success(f"💾 تم حفظ النتائج في ملف: {filename}")
                        
                        # عرض محتوى الملف
                        with open(filename, "r", encoding="utf-8") as f:
                            st.download_button(
                                label="📥 تحميل النتائج",
                                data=f.read(),
                                file_name=filename,
                                mime="text/plain"
                            )
                
                except Exception as e:
                    st.error(f"❌ خطأ في تحليل الصورة: {e}")
                    st.info("""
                    **اقتراحات للتصحيح:**
                    1. تحقق من أن الصورة ملونة (RGB)
                    2. جرب صورة بحجم مختلف
                    3. تأكد من أن النموذج متوافق مع نوع الصورة
                    """)
        
        else:
            st.markdown('<div class="info-box">⬅️ يرجى تحميل صورة لرؤية النتائج</div>', unsafe_allow_html=True)
            
            # صورة توضيحية
            st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400", 
                    caption="📤 انتظر تحميل الصورة...", 
                    use_column_width=True)
    
    # قسم المساعدة
    st.markdown("---")
    with st.expander("❓ المساعدة والدعم"):
        st.markdown("""
        ### كيفية الاستخدام:
        1. اختر طريقة تحميل الصورة من الشريط الجانبي
        2. انتظر حتى يتم تحميل النموذج
        3. شاهد نتائج التحليل
        
        ### نصائح:
        - استخدم صور واضحة وجيدة الإضاءة
        - يمكنك ضبط حد الثقة من الشريط الجانبي
        - للنتائج الأفضل، استخدم صور بحجم 224×224 بكسل
        
        ### معلومات تقنية:
        - النموذج: TensorFlow/Keras
        - المعالجة: Pillow
        - الاستضافة: Streamlit Cloud
        
        ### التواصل والدعم:
        - في حالة وجود مشاكل، حاول تحديث الصفحة
        - تأكد من اتصال الإنترنت
        - تحقق من رابط النموذج
        """)

# تشغيل التطبيق
if __name__ == "__main__":
    main()
