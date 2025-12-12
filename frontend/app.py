import streamlit as st
import requests
from PIL import Image
import io
import time

# --- Configuration (Unchanged) ---
API_URL = "http://127.0.0.1:8000/api/v1/predict/" 
COLOR_MAP = {"Normal": "green", "Benign": "orange", "Malignant": "red"}
LABEL_DISPLAY_MAP = {"Normal": "✅ NORMAL", "Benign": "🟡 BENIGN", "Malignant": "❌ MALIGNANT"}

# --- Background Image URL (Using a safe, subtle local color instead of URL) ---
# NOTE: If you still want a background image, ensure the URL is valid and accessible.
# Otherwise, using a simple fixed background color is safer.
BACKGROUND_IMAGE_URL = "" # Emptying the URL for now to prevent breakage

st.set_page_config(page_title="Breast Cancer AI Diagnostic", page_icon="🎗️", layout="wide")

# --- Custom CSS for Branding, Professional Look, and BACKGROUND ---
st.markdown(
    f"""
    <style>
    /* FIX: Simple background style to prevent URL issues */
    .stApp {{
        background-color: #f7f7f7; /* Light gray background for contrast */
        background-image: url("{BACKGROUND_IMAGE_URL}"); /* This line is now safe/inactive */
        background-size: cover;
        background-attachment: fixed;
    }}
    
    :root {{--primary-color-brand: #e91e63;}}
    .main-title {{color: var(--primary-color-brand); text-align: center; font-size: 3rem; font-weight: bold;}}
    .subtitle {{text-align: center; color: #555; font-size: 1.2rem; margin-bottom: 2rem;}}
    .stButton>button {{background-color: var(--primary-color-brand); color: white; border-radius: 10px; width: 100%; height: 3em; font-size: 1.2em; transition: background-color 0.3s;}}
    .stButton>button:hover {{background-color: #d81b60;}}
    [data-testid="stMetricValue"] {{font-size: 2rem;}}
    </style>
""", unsafe_allow_html=True)


# --- 1. SIDEBAR (Context & Branding) ---
with st.sidebar:
    # FIX: Replacing the problematic st.image with a clear Title/Emoji
    st.title(" Project Context") 
    
    st.subheader("Developers ()")
    st.info("Sapir Baruch & Eden Oren")
    
    st.subheader("Model")
    st.markdown("Currently using **ResNet-50** (via Transfer Learning) for robust classification.")
    
    st.subheader("Disclaimer")
    st.warning("This tool is for educational purposes and internal review only. **Always consult a medical professional** for any health-related concerns.")
    

# --- 2. MAIN HEADER ---
st.markdown('<div class="main-title"> AI Breast Cancer Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a mammogram or ultrasound image for instant analysis.</div>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state for button management
if 'last_button_pressed' not in st.session_state:
    st.session_state['last_button_pressed'] = False


# --- 3. WORKFLOW LAYOUT (Step 1 & Step 2) ---
col_upload, col_result = st.columns([1, 1])
uploaded_file = None

with col_upload:
    st.header("1️⃣ Upload Image")
    st.info("Upload a clear PNG, JPG, or JPEG file to start the diagnosis process.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.subheader("Uploaded Image:")
        st.image(uploaded_file, caption="Image Ready for Analysis", use_column_width=True)
        st.markdown("---")
        
        if st.button("🔍 Analyze Image"):
            st.session_state['last_button_pressed'] = True
            st.experimental_rerun() 


# --- 4. RESULT LOGIC (Step 2) ---
if uploaded_file is not None and st.session_state.get('last_button_pressed'):
    
    with col_result:
        st.header("2️⃣ Diagnosis Result")
        
        with st.spinner("Analyzing image using ResNet-50 server..."):
            time.sleep(0.5) 
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    prediction_label = data['prediction']
                    probabilities = data['probabilities'] 
                    
                    confidence_value = max(probabilities.values())
                    confidence_display = f"{confidence_value*100:.1f}%"
                    
                    if prediction_label == "Normal":
                        st.balloons()
                        st.success(f"### FINAL: {LABEL_DISPLAY_MAP[prediction_label]}")
                        st.markdown(f":{COLOR_MAP[prediction_label]}[The model suggests a very **low probability** of malignancy.]")
                    elif prediction_label == "Benign":
                        st.warning(f"### FINAL: {LABEL_DISPLAY_MAP[prediction_label]}")
                        st.markdown(f":{COLOR_MAP[prediction_label]}[A mass was detected, classified as **Non-Malignant** (Benign).]")
                    else:
                        st.error(f"### FINAL: {LABEL_DISPLAY_MAP[prediction_label]}")
                        st.markdown(f":{COLOR_MAP[prediction_label]}[**CRITICAL ALERT:** The model indicates a **Malignant** classification. Seek immediate specialist consultation.]")

                    st.markdown("---")
                    
                    metric1, metric2 = st.columns(2)
                    
                    with metric1:
                        st.metric(label="Predicted Classification", value=prediction_label, delta_color=COLOR_MAP[prediction_label])
                    
                    with metric2:
                        st.metric(label="Confidence Score", value=confidence_display)

                    with st.expander("Detailed Probability Breakdown (Model Raw Output)", expanded=False):
                        st.caption(f"Model used: {data.get('model_used', 'ResNet50')}")
                        
                        def get_prob(label):
                            return probabilities.get(label, 0.0)

                        prob_normal = get_prob("Normal")
                        st.write(f"🟢 **Normal:** {prob_normal*100:.1f}%")
                        st.progress(int(prob_normal * 100))
                        
                        prob_benign = get_prob("Benign")
                        st.write(f"🟠 **Benign:** {prob_benign*100:.1f}%")
                        st.progress(int(prob_benign * 100))
                        
                        prob_malignant = get_prob("Malignant")
                        st.write(f"🔴 **Malignant:** {prob_malignant*100:.1f}%")
                        st.progress(int(prob_malignant * 100))
                        

                else:
                    error_data = response.json()
                    st.exception(f"Server Error (Code {response.status_code}): {error_data.get('detail', response.text)}")
                    st.caption("Please check the backend terminal for details.")

            except requests.exceptions.ConnectionError:
                st.exception(f"Connection Error! Ensure the FastAPI server is running at: {API_URL}")
            except Exception as e:
                st.exception(f"An unexpected error occurred: {e}")
                
    st.session_state['last_button_pressed'] = False