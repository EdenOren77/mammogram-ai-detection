import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="Breast Cancer AI Diagnostic",
    page_icon="🎗️",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title {
        color: #e91e63; 
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #e91e63;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"> AI Breast Cancer Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a mammogram/ultrasound image for instant analysis based on ResNet50</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        
        with st.spinner("Sending to ResNet50 server..."):
            try:
                files = {"file": uploaded_file.getvalue()}

                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    
                    prediction = data["prediction"]
                    confidence = data["confidence"]
                    probabilities = data["probabilities"]

                    st.success("Analysis Complete!")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Diagnosis", value=prediction)
                    
                    with col2:
                        st.metric(label="Confidence", value=confidence)

                    st.write("---")
                    st.subheader("Probability Breakdown")
                    
                    # Normal
                    st.write(f"🟢 Normal: {probabilities['normal']*100:.1f}%")
                    st.progress(int(probabilities['normal'] * 100))
                    
                    # Benign
                    st.write(f"🟠 Benign: {probabilities['benign']*100:.1f}%")
                    st.progress(int(probabilities['benign'] * 100))
                    
                    # Malignant
                    st.write(f"🔴 Malignant: {probabilities['malignant']*100:.1f}%")
                    st.progress(int(probabilities['malignant'] * 100))

                else:
                    st.error(f"Server Error: {response.text}")

            except Exception as e:
                st.error(f"Connection Error! Is the server running? \nDetails: {e}")