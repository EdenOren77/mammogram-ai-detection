import streamlit as st

st.set_page_config(
    page_title="Mammogram AI",
    page_icon="🎗️",
    layout="centered",
)

st.markdown(
    """
<style>
/* App background */
.stApp {
    background: radial-gradient(
        circle at top,
        rgba(255, 182, 193, 0.32),
        rgba(255, 235, 242, 0.6) 40%,
        rgba(255, 255, 255, 1) 75%
    );
}


/* Content width & spacing */
.block-container {
    padding-top: 2.5rem;
    max-width: 900px;
}

/* Typography */
.h1 {
    font-size: 2.05rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    margin: 0;
}

.sub {
    font-size: 1rem;
    color: rgba(0,0,0,0.62);
    margin-top: .35rem;
    margin-bottom: 1.4rem;
}

/* Very soft card */
.card {
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 16px;
    padding: 16px 18px;
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(6px);
}
</style>
""",
    unsafe_allow_html=True,
)


# Header
st.markdown('<div class="pill">Demo</div><div class="pill">ResNet50</div>', unsafe_allow_html=True)
st.markdown('<h1 class="h1">Mammogram AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Upload a mammogram or ultrasound image and get a probability breakdown (Normal / Benign / Malignant).</div>',
    unsafe_allow_html=True,
)

# Main card
st.markdown(
    """
<div class="card">
<b>Start here</b><br/>
Open <b>Analyze</b> from the sidebar, upload an image, and click <b>Analyze Image</b>.<br/><br/>
<b>What you’ll get</b><br/>
• Predicted label + confidence<br/>
• Probability bars that always sum to 100%
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
st.caption("🎗️ Educational demo only — not a medical diagnosis. Always consult a medical professional.")
