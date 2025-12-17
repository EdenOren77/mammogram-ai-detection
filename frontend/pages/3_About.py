import streamlit as st
from ui.theme import apply_theme

apply_theme()

st.markdown("## About")
st.markdown(
    """
This UI is designed to feel like a real MedTech product:
- Clean layout & typography
- Robust backend integration
- Clear states (idle / analyzing / error / result)

**Disclaimer:** Not a medical diagnosis.
"""
)
