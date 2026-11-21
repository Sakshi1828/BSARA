"""BSARA v2 demo: image-level bloodstain pattern classification.

Run:  streamlit run app.py
Needs models/bsara_cls_v2.pt (python -m bsara.train_final, or download it
from the GitHub Release).

This is a research prototype trained on three small laboratory datasets. It is
NOT a forensic tool; see the README "Limitations" section.
"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "bsara_cls_v2.pt"

CLASS_INFO = {
    "drip": "Passive drip: drops falling under gravity.",
    "impact_beating": "Impact spatter: a blood source struck by a blunt object.",
    "gunshot": "Gunshot backspatter: fine mist travelling back toward the firearm.",
}

st.set_page_config(page_title="BSARA v2", layout="wide")
st.title("BSARA: Bloodstain Pattern Classification (research prototype)")
st.warning(
    "**Not for casework.** Trained on 1,101 lab images from three different datasets, one per "
    "class. Cross-validation shows the classes can be told apart from background and photography "
    "cues alone, even with the blood removed, so a prediction here does **not** "
    "mean the model understood the stain. See the README for the full evaluation."
)


@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error(f"Model not found at `{MODEL_PATH.relative_to(ROOT)}`. "
             "Run `python -m bsara.train_final` or download it from the GitHub Release.")
    st.stop()

model = load_model()
col1, col2 = st.columns(2)
with col1:
    up = st.file_uploader("Upload a bloodstain image", type=["jpg", "jpeg", "png"])
    if up:
        image = Image.open(up).convert("RGB")
        st.image(image, use_container_width=True)

with col2:
    if up:
        res = model.predict(np.ascontiguousarray(np.array(image)[:, :, ::-1]), imgsz=224, verbose=False)[0]
        probs = res.probs.data.cpu().numpy()
        order = probs.argsort()[::-1]
        top = model.names[int(order[0])]
        st.subheader(f"Most likely: {top.replace('_', ' ')}  ({probs[order[0]]:.0%})")
        st.caption(CLASS_INFO.get(top, ""))
        for i in order:
            st.progress(float(probs[i]), text=f"{model.names[int(i)]}: {probs[i]:.1%}")
        st.caption("Scores are softmax outputs, not calibrated probabilities.")
    else:
        st.info("Upload an image to classify it.")
