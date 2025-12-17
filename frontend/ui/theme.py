import streamlit as st

def apply_theme():
    st.set_page_config(
        page_title="Breast Cancer AI",
        page_icon="🎗️",
        layout="wide",
    )

    st.markdown(
        """
<style>
:root{
  --bg-main: #FAFAFC;
  --bg-surface: #FFFFFF;
  --pink-primary: #E84C88;
  --pink-soft: #F7C1D9;
  --text-main: #1F2937;
  --text-muted: #6B7280;
  --border-subtle: rgba(0,0,0,0.08);
}

/* App background */
html, body, [data-testid="stAppViewContainer"]{
  background: linear-gradient(
    180deg,
    #FAFAFC 0%,
    #FDF0F6 100%
  );
  color: var(--text-main);
}

/* Main container */
.block-container{
  max-width: 1200px;
  padding-top: 1.5rem;
}

/* Cards */
.card{
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 18px;
  padding: 20px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.06);
}

/* Result accent */
.result-card{
  border-left: 4px solid var(--pink-primary);
}

/* Text */
.subtle{
  color: var(--text-muted);
  font-size: 13px;
}

/* Badge */
.badge{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(232,76,136,0.12);
  color: var(--pink-primary);
  border: 1px solid rgba(232,76,136,0.25);
  font-size: 13px;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"]{
  border-radius: 18px !important;
  border: 1px dashed rgba(232,76,136,0.4) !important;
  background: rgba(232,76,136,0.04) !important;
}

/* Fix uploader text color */
section[data-testid="stFileUploaderDropzone"] * {
  color: var(--text-main) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
