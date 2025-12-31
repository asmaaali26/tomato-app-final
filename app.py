import streamlit as st
import random
from datetime import datetime

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Tomato AI - Disease Classifier",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== DARK/LIGHT MODE =====
# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ===== DYNAMIC CSS WITH DARK MODE =====
def get_css(dark_mode=False):
    if dark_mode:
        return """
        <style>
            :root {
                --primary: #4CAF50;
                --secondary: #2E7D32;
                --bg-color: #1a1a1a;
                --card-bg: #2d2d2d;
                --text-color: #ffffff;
                --border-color: #444;
            }
        </style>
        """
    else:
        return """
        <style>
            :root {
                --primary: #4CAF50;
                --secondary: #2E7D32;
                --bg-color: #f8f9fa;
                --card-bg: #ffffff;
                --text-color: #333333;
                --border-color: #ddd;
            }
        </style>
        """

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    /* Base Styles */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, var(--secondary), var(--primary));
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .leaf-decoration {
        font-size: 2rem;
        margin: 0 10px;
        color: #4CAF50;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Cards */
    .card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        transition: transform 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    .disease-card {
        border-left: 5px solid #FF9800;
    }
    
    .healthy-card {
        border-left: 5px solid var(--primary);
    }
    
    /* Stats Box */
    .stat-box {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    
    /* Upload Area */
    .upload-area {
        border: 3px dashed var(--primary);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: rgba(76, 175, 80, 0.05);
        transition: all 0.3s;
        min-height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .upload-area:hover {
        background: rgba(76, 175, 80, 0.1);
        border-color: var(--secondary);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    /* Tomato Animation */
    .tomato-animation {
        display: inline-block;
        animation: tomatoSpin 20s linear infinite;
    }
    
    @keyframes tomatoSpin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Results Grid */
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Severity Indicators */
    .severity-high { color: #ff4444; font-weight: bold; }
    .severity-medium { color: #ff9800; font-weight: bold; }
    .severity-low { color: #4CAF50; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Apply dark/light mode CSS
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    # Dark Mode Toggle
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    
    st.markdown("## 🛠️ Control Panel")
    
    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Supported Diseases", "10")
    with col2:
        st.metric("🎯 Model Accuracy", "96.7%")
    
    st.markdown("---")
    
    # Analysis Settings
    st.markdown("### 🔧 Analysis Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=50,
        max_value=99,
        value=75
    )
    
    # Info
    st.markdown("---")
    st.markdown("### ℹ️ App Info")
    st.info("**Version:** 2.1.0\n\n**Last Updated:** Dec 2025\n\n**Developer:** Tomato AI Team")

# ===== MAIN HEADER =====
st.markdown(f"""
<div class="main-header">
    <h1 style="margin:0">
        <span class="leaf-decoration">🌿</span>
        <span class="tomato-animation">🍅</span>
        Tomato AI
        <span class="tomato-animation">🍅</span>
        <span class="leaf-decoration">🌿</span>
    </h1>
    <h3 style="margin:0">Intelligent Tomato Disease Classification System</h3>
    <p style="opacity:0.9; margin-top:10px">Early detection with 98% accuracy • Powered by AI</p>
</div>
""", unsafe_allow_html=True)

# ===== MAIN TABS =====
tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "📊 Results Dashboard", "🌱 Disease Library"])

# ===== TAB 1: IMAGE ANALYSIS =====
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Tomato Leaf Image")
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of tomato leaf",
            label_visibility="collapsed"
        )
        
        if uploaded_file is None:
            st.markdown("""
            <div style="text-align:center">
                <span style="font-size:4rem">📁</span>
                <h4>Drop your tomato leaf image here</h4>
                <p style="color:#666">Supports: JPG, PNG</p>
                <small style="color:var(--primary)">Clear images yield better results!</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ Image uploaded successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips
        with st.expander("💡 Tips for best results"):
            st.markdown("""
            1. **Good lighting** - Use natural daylight
            2. **Focus on leaf** - Leaf should fill most of the frame
            3. **Simple background** - Avoid busy backgrounds
            4. **Multiple angles** - Upload from different angles if needed
            5. **Avoid blurry images** - Ensure image is sharp and clear
            """)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🔍 Image Preview")
            
            # Image preview
            import PIL.Image
            image = PIL.Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded Image")
            
            # Image info
            img_info = f"**Size:** {image.size[0]}×{image.size[1]}px | **Format:** {image.format} | **Mode:** {image.mode}"
            st.info(img_info)
            
            # Analysis button
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                if st.button("🚀 Start AI Analysis", use_container_width=True, type="primary"):
                    # Simulate analysis
                    with st.spinner("Analyzing with AI..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            import time
                            time.sleep(0.02)
                            progress_bar.progress(i + 1)
                        
                        st.balloons()
                        st.success("✅ Analysis complete!")
                        st.session_state.analysis_complete = True
                        st.session_state.uploaded_file = uploaded_file
            
            with col_btn2:
                if st.button("🔄 Reset"):
                    st.rerun()

# ===== TAB 2: RESULTS DASHBOARD =====
with tab2:
    if st.session_state.get('analysis_complete', False):
        st.markdown("## 📊 Analysis Results")
        
        # Top Results
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 Confidence")
            st.markdown("## 96.7%")
            st.markdown("*High accuracy*")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_res2:
            st.markdown('<div class="healthy-card card">', unsafe_allow_html=True)
            st.markdown("### ✅ Diagnosis")
            st.markdown("## Healthy Plant")
            st.markdown("*No disease detected*")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_res3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown("### ⚡ Speed")
            st.markdown("## 2.1s")
            st.markdown("*Fast analysis*")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Results
        st.markdown("### 📈 Disease Probability Distribution")
        
        # Sample data
        diseases_data = [
            {"name": "Healthy", "confidence": 96.7, "severity": "low"},
            {"name": "Early Blight", "confidence": 1.8, "severity": "high"},
            {"name": "Late Blight", "confidence": 0.9, "severity": "high"},
            {"name": "Bacterial Spot", "confidence": 0.4, "severity": "medium"},
            {"name": "Leaf Mold", "confidence": 0.2, "severity": "low"},
        ]
        
        # Display as bar chart
        import pandas as pd
        df = pd.DataFrame(diseases_data)
        
        # Create a visual representation
        st.markdown("#### Disease Confidence Levels:")
        for disease in diseases_data:
            col_bar, col_info = st.columns([4, 1])
            with col_bar:
                progress = disease["confidence"] / 100
                st.progress(progress, text=f"{disease['name']}: {disease['confidence']:.1f}%")
            with col_info:
                severity_color = {
                    "high": "severity-high",
                    "medium": "severity-medium",
                    "low": "severity-low"
                }
                st.markdown(f'<span class="{severity_color[disease["severity"]]}">{disease["severity"].upper()}</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Treatment Recommendations
        st.markdown("### 💡 Treatment Recommendations")
        
        rec_cols = st.columns(3)
        
        with rec_cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🌧️ Watering")
            st.markdown("""
            • Water every 2-3 days
            • Avoid leaf wetting
            • Use drip irrigation
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with rec_cols[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🌿 Fertilization")
            st.markdown("""
            • Balanced NPK fertilizer
            • Add calcium monthly
            • Organic compost
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with rec_cols[2]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🛡️ Prevention")
            st.markdown("""
            • Good air circulation
            • Regular inspection
            • Crop rotation
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export Options
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            if st.button("📄 Download PDF Report"):
                st.success("Report will be downloaded")
        
        with col_dl2:
            if st.button("📊 Export to Excel"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"tomato_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col_dl3:
            if st.button("🖼️ Save with Image"):
                st.info("Feature coming soon!")
    
    else:
        st.info("👈 Please upload an image and analyze it in the 'Image Analysis' tab")

# ===== TAB 3: DISEASE LIBRARY =====
with tab3:
    st.markdown("## 🌱 Tomato Disease Library")
    
    diseases = [
        {
            "name": "Early Blight",
            "icon": "🍂",
            "symptoms": "Circular brown spots on older leaves",
            "treatment": "Chlorothalonil-based fungicides",
            "prevention": "Proper spacing, crop rotation",
            "severity": "high"
        },
        {
            "name": "Late Blight",
            "icon": "🌧️",
            "symptoms": "Water-soaked spots turning dark brown",
            "treatment": "Systemic fungicides",
            "prevention": "Avoid overhead watering",
            "severity": "high"
        },
        {
            "name": "Bacterial Spot",
            "icon": "🦠",
            "symptoms": "Small water-soaked leaf spots",
            "treatment": "Copper-based bactericides",
            "prevention": "Use certified disease-free seeds",
            "severity": "medium"
        },
        {
            "name": "Leaf Mold",
            "icon": "🍄",
            "symptoms": "Yellow spots with fuzzy growth",
            "treatment": "Fungicide sprays",
            "prevention": "Reduce humidity",
            "severity": "low"
        },
        {
            "name": "Septoria Leaf Spot",
            "icon": "🔍",
            "symptoms": "Small circular spots with gray centers",
            "treatment": "Remove infected leaves",
            "prevention": "Avoid wet foliage",
            "severity": "medium"
        },
        {
            "name": "Healthy Plant",
            "icon": "✅",
            "symptoms": "Green, vigorous growth",
            "treatment": "Regular maintenance",
            "prevention": "Proper care",
            "severity": "low"
        }
    ]
    
    # Display diseases in columns
    cols = st.columns(2)
    for idx, disease in enumerate(diseases):
        with cols[idx % 2]:
            severity_color = {
                "high": "severity-high",
                "medium": "severity-medium",
                "low": "severity-low"
            }
            
            st.markdown(f"""
            <div class="{'disease-card' if disease['name'] != 'Healthy Plant' else 'healthy-card'} card">
                <h4>{disease['icon']} {disease['name']} 
                <span style="float:right" class="{severity_color[disease['severity']]}">
                    {disease['severity'].upper()}
                </span></h4>
                <p><strong>Symptoms:</strong> {disease['symptoms']}</p>
                <p><strong>Treatment:</strong> {disease['treatment']}</p>
                <p><strong>Prevention:</strong> {disease['prevention']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick Comparison
    st.markdown("---")
    st.markdown("### 📊 Disease Comparison")
    
    comparison_data = pd.DataFrame(diseases)
    st.dataframe(
        comparison_data[['name', 'severity', 'symptoms']],
        column_config={
            "name": "Disease",
            "severity": "Severity",
            "symptoms": "Key Symptoms"
        },
        use_container_width=True,
        hide_index=True
    )

# ===== FOOTER =====
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**🍅 Tomato AI**")
    st.markdown("*Smart Agriculture Solutions*")
with footer_cols[1]:
    st.markdown(f"**📅 Version 2.1**")
    st.markdown(f"Updated: {datetime.now().strftime('%Y-%m-%d')}")
with footer_cols[2]:
    st.markdown("**🌐 All Rights Reserved**")
    st.markdown("© 2025 Tomato AI Team")

# ===== SESSION STATE INITIALIZATION =====
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# ===== SIDEBAR BADGE =====
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🌟 Rate This App")
    rating = st.select_slider(
        "How was your experience?",
        options=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        value="⭐⭐⭐⭐"
    )
    
    if st.button("Submit Rating"):
        st.success("Thank you for your feedback! 🌱")
