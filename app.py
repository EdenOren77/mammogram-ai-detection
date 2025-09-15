import streamlit as st
import torch
import cv2
import numpy as np
from src.model import SimpleCNN
import torch.nn.functional as F
from PIL import Image


st.set_page_config(page_title="Breast Cancer Detection AI", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #fff0f5;
    }
    .pink-title {
        font-size: 36px;
        font-weight: bold;
        color: #e75480;
        text-align: center;
    }
    .subtext {
        font-size: 16px;
        color: #c71585;
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        font-size: 14px;
        text-align: center;
        color: gray;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# כותרת
st.markdown('<div class="pink-title">💗 Breast Cancer Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">By Eden Oren – Computer Science Student | Educational project only</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Early detection saves lives. This tool is not for medical use. Please consult a doctor.</div>', unsafe_allow_html=True)

# כפתור מעבר
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

if not st.session_state.show_upload:
    if st.button("💗 Let's Get Started!"):
        st.session_state.show_upload = True
    st.stop()


uploaded_file = st.file_uploader("Upload a breast ultrasound image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = image.resize((224, 224))
    image_np = np.array(image)
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0


    model = SimpleCNN()
    model.load_state_dict(torch.load("saved_models/mammogram_cnn.pth", map_location=torch.device('cpu')))
    model.eval()


    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    labels_map = {0: "Normal", 1: "Benign", 2: "Malignant"}
    colors = {0: "#32CD32", 1: "#FFA07A", 2: "#FF1493"}


    st.markdown(f"<h2 style='color:{colors[predicted_label]}; text-align:center;'>Prediction: {labels_map[predicted_label]}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>Probabilities: {probabilities.numpy()}</p>", unsafe_allow_html=True)


st.markdown('<div class="footer">🩷 Developed with love by Eden Oren · Powered by PyTorch & Streamlit</div>', unsafe_allow_html=True)