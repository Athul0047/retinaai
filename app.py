"""Local Streamlit interface for the educational RetinaAI prototype."""

from __future__ import annotations

import hashlib
import io
import os

# Streamlit usage telemetry is unnecessary for this local educational app.
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.prediction import PredictionError, RetinaPredictor


GRADE_DESCRIPTIONS = {
    0: "No diabetic retinopathy findings are assigned to this category.",
    1: "Mild nonproliferative DR. Small vessel changes, such as microaneurysms, may be present.",
    2: "Moderate nonproliferative DR. Retinal changes are more developed than the mild category.",
    3: "Severe nonproliferative DR. More extensive vessel changes may be present.",
    4: "Proliferative DR. Abnormal new blood vessels may develop in this advanced stage.",
}


@st.cache_resource(show_spinner=False)
def get_predictor() -> RetinaPredictor:
    """Cache model loading between Streamlit reruns and user interactions."""
    return RetinaPredictor()


def validate_upload(filename: str, contents: bytes) -> tuple[Image.Image | None, str | None]:
    """Check the filename extension and verify that the bytes contain an image."""
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return None, "Choose a JPG, JPEG, or PNG image."
    if not contents:
        return None, "The selected file is empty. Choose a different image."

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
        with Image.open(io.BytesIO(contents)) as image:
            return image.convert("RGB"), None
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None, "This file could not be read as a valid image. Try another JPG, JPEG, or PNG."


def main() -> None:
    st.set_page_config(
        page_title="RetinaAI | DR learning prototype",
        page_icon="👁️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
        .hero {padding: 1.6rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#eef6ff,#f2f0ff); margin-bottom: 1.2rem;}
        .hero p {font-size: 1.05rem; margin-bottom: 0; color: #334155;}
        .result-card {padding: 1.2rem; border: 1px solid #dbe4ef; border-radius: 14px; background: #fbfdff;}
        .grade-number {font-size: 2.3rem; font-weight: 700; color: #17365d; margin: 0;}
        .muted {color: #526174; font-size: .92rem;}
        @media (max-width: 700px) { .block-container {padding: 1rem;} .hero {padding: 1.2rem;} }
        </style>
        <div class="hero">
          <h1>RetinaAI</h1>
          <p>Explore a five-grade diabetic retinopathy model with a retinal fundus image.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "**Student educational prototype only.** This model is not clinically validated "
        "and must not be used for diagnosis, screening decisions, or treatment. "
        "A model score is not medical certainty or a substitute for an eye-care professional."
    )

    st.subheader("Try an image")
    st.write("Select a retinal fundus image. The image stays in this local app session.")
    uploaded = st.file_uploader(
        "Upload a JPG, JPEG, or PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Use a readable retinal fundus photograph. No image is sent to a hosted prediction service.",
    )

    image = None
    image_bytes = b""
    image_key = ""
    validation_error = None
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        image_key = hashlib.sha256(image_bytes).hexdigest()
        image, validation_error = validate_upload(uploaded.name, image_bytes)

    preview_col, action_col = st.columns([1.1, 0.9], gap="large")
    with preview_col:
        st.markdown("#### Image preview")
        if validation_error:
            st.error(validation_error)
        elif image is not None:
            st.image(image, caption=uploaded.name, use_container_width=True)
        else:
            st.info("Your selected image will appear here.")

    with action_col:
        st.markdown("#### Prediction")
        st.markdown(
            '<p class="muted">Inference runs only after you select Predict. '
            "The model is loaded once and reused during this app session.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Predict", type="primary", disabled=image is None, use_container_width=True):
            try:
                with st.spinner("Loading the local model and analyzing the image…"):
                    prediction = get_predictor().predict(image)
                st.session_state["retinaai_prediction"] = {
                    "image_key": image_key,
                    "result": prediction,
                    "error": None,
                }
            except (PredictionError, OSError, RuntimeError) as exc:
                st.session_state["retinaai_prediction"] = {
                    "image_key": image_key,
                    "result": None,
                    "error": str(exc),
                }

        saved = st.session_state.get("retinaai_prediction")
        if saved and saved.get("image_key") == image_key and image is not None:
            if saved.get("error"):
                st.error(f"Prediction could not be completed: {saved['error']}")
            elif saved.get("result"):
                result = saved["result"]
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"<p class='muted'>Predicted category · Grade {result['grade_index']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='grade-number'>{result['class_name']}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption(GRADE_DESCRIPTIONS[result["grade_index"]])
                st.markdown("**Model scores for all five classes**")
                st.caption("Scores are softmax outputs from this model; they are not calibrated medical certainty.")
                for grade, (class_name, score) in enumerate(result["probabilities"].items()):
                    st.write(f"Grade {grade} · {class_name} — {score:.1%}")
                    st.progress(max(0.0, min(1.0, score)))

    with st.expander("Learn about the five grade labels"):
        for grade, description in GRADE_DESCRIPTIONS.items():
            st.markdown(f"**Grade {grade} · {('No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR')[grade]}**  \n{description}")
        st.caption(
            "These brief descriptions are educational summaries of common category labels, not image findings. "
            "Learn more from the [National Eye Institute](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy)."
        )

    st.divider()
    st.caption(
        "RetinaAI is a student educational prototype trained on APTOS 2019. "
        "It is not clinically validated and is not for diagnosis, screening decisions, or treatment."
    )


if __name__ == "__main__":
    main()
