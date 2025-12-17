import streamlit as st
from PIL import Image
from ui.theme import apply_theme
from ui.components import card_start, card_end, header, result_badge, meaning_text
from api.client import predict, BackendError

apply_theme()

DEFAULT_API_URL = "http://127.0.0.1:5001/predict"

with st.sidebar:
    st.markdown("### Settings")
    api_url = st.text_input("Backend URL", value=DEFAULT_API_URL)
    timeout_s = st.slider("Timeout (seconds)", 3, 60, 20)
    st.markdown("---")
    st.caption("This tool is a demo and **not** a medical diagnosis.")

header("Analyze", "Upload an image and get a prediction with probability breakdown.")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    card_start()
    st.markdown("### Upload")
    uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown('<div class="subtle">Tip: clearer input → more stable output.</div>', unsafe_allow_html=True)
    card_end()

with right:
    card_start()
    st.markdown("### Result")

    if not uploaded:
        st.info("Upload an image to begin.")
        card_end()
    else:
        # preview
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Preview", use_container_width=True)
        except Exception:
            st.warning("Could not preview the image, but you can still analyze it.")

        analyze = st.button("🚀 Analyze", use_container_width=True)

        if analyze:
            with st.spinner("Analyzing…"):
                try:
                    res, latency_ms = predict(
                        api_url=api_url,
                        file_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        timeout_s=timeout_s,
                    )

                    result_badge(res.label, res.confidence_pct, latency_ms)
                    st.progress(res.confidence_pct)
                    meaning_text(res.label)

                    st.markdown("---")
                    st.markdown("### Probability breakdown")

                    for k in ["Normal", "Benign", "Malignant"]:
                        v = res.probabilities_pct.get(k, 0)
                        emoji = "🟢" if k == "Normal" else ("🟠" if k == "Benign" else "🔴")
                        st.write(f"{emoji} **{k}** — {v}%")
                        st.progress(v)

                except BackendError as e:
                    st.error(f"Backend error: {e}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        card_end()
