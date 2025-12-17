import streamlit as st

def color_for(label: str) -> str:
    if label == "Normal":
        return "#22c55e"
    if label == "Benign":
        return "#f59e0b"
    if label == "Malignant":
        return "#ef4444"
    return "#a1a1aa"

def card_start():
    st.markdown('<div class="card">', unsafe_allow_html=True)

def card_end():
    st.markdown("</div>", unsafe_allow_html=True)

def header(title: str, subtitle: str):
    st.markdown(f"## {title}")
    st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)

def result_badge(label: str, confidence_pct: int, latency_ms: int):
    c = color_for(label)
    st.markdown(
        f"""
<div class="badge">
  <span class="dot" style="background:{c}"></span>
  <span><b>{label}</b></span>
  <span style="opacity:0.7;">|</span>
  <span><b>{confidence_pct}%</b> confidence</span>
  <span style="opacity:0.7;">|</span>
  <span>{latency_ms}ms</span>
</div>
""",
        unsafe_allow_html=True,
    )

def meaning_text(label: str):
    if label == "Normal":
        st.info("No suspicious finding detected by the model. This is not a medical diagnosis.")
    elif label == "Benign":
        st.warning("A non-cancerous pattern is more likely according to the model. Not a diagnosis.")
    elif label == "Malignant":
        st.error("A potentially cancerous pattern is more likely according to the model. Not a diagnosis.")
    else:
        st.info("The model returned an unknown label. Check the backend response.")
