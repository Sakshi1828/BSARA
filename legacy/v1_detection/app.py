import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import datetime

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── WHAT IS THIS FILE? ─────────────────────────────────────────────
# This is the main web application file
# Streamlit reads this file top to bottom and builds a webpage from it
# Every time the user interacts (uploads image, clicks button),
# Streamlit reruns this entire file from top to bottom

# ── PAGE CONFIGURATION ─────────────────────────────────────────────
# This must be the FIRST streamlit command in the file
# It sets the browser tab title and page layout
st.set_page_config(
    page_title="BSARA — Bloodstain Pattern Analysis",
    page_icon="🔬",
    layout="wide"  # wide = full browser width instead of narrow centered
)

# ── LOAD THE MODEL ─────────────────────────────────────────────────
# @st.cache_resource means:
# "Load this once and keep it in memory"
# Without this, the model would reload every time user clicks anything
# Loading a model takes ~2 seconds — we don't want that every click
@st.cache_resource
def load_model():
    # Find the best weights from your training run
    # Change bsara_v13 to whatever your run folder is named
    model_path = os.path.join(ROOT, "runs", "bsara_v13", "weights", "best.pt")
    model = YOLO(model_path)
    return model

model = load_model()

# ── CLASS INFORMATION ──────────────────────────────────────────────
# Dictionary storing information about each bloodstain class
# This is used to display descriptions in the forensic report
CLASS_INFO = {
    "drip": {
        "color": (59, 130, 246),   # Blue in BGR
        "description": "Passive drip pattern — blood fell vertically due to gravity. Indicates a stationary or slow-moving blood source.",
        "severity": "Low velocity event"
    },
    "impact_beating": {
        "color": (239, 68, 68),    # Red in BGR
        "description": "Impact spatter — caused by an external force striking a blood source. Pattern indicates direction and force of impact.",
        "severity": "High velocity blunt force event"
    },
    "gunshot": {
        "color": (245, 158, 11),   # Orange in BGR
        "description": "Gunshot backspatter — fine mist pattern caused by bullet impact. Droplets travel opposite to bullet direction.",
        "severity": "High velocity projectile event"
    }
}

# ── APP HEADER ─────────────────────────────────────────────────────
# st.title creates a large heading
# st.markdown lets you write formatted text (bold, italic etc)
st.title("🔬 BSARA — Bloodstain Pattern Analysis & Reconstruction AI")
st.markdown("**Forensic AI System** | Automated bloodstain detection and classification")
st.markdown("---")  # horizontal line

# ── SIDEBAR ────────────────────────────────────────────────────────
# The sidebar is a panel on the left side of the page
# Good for settings and information that doesn't change
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    # Confidence threshold slider
    # This controls how confident the model must be before showing a detection
    # 0.25 = show detections the model is 25%+ confident about
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower = more detections but possibly wrong. Higher = fewer but more accurate."
    )
    
    st.markdown("---")
    st.header("📋 About")
    st.markdown("""
    **BSARA** uses YOLOv8 computer vision
    to automatically detect and classify
    bloodstain patterns.
    
    **Classes detected:**
    - 🔵 Drip patterns
    - 🔴 Impact beating spatter  
    - 🟠 Gunshot backspatter
    """)

# ── MAIN CONTENT ───────────────────────────────────────────────────
# Split the page into 2 columns
# col1 = left side (upload + original image)
# col2 = right side (analyzed image + results)
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    
    # File uploader widget
    # type= restricts what files can be uploaded
    uploaded_file = st.file_uploader(
        "Upload a crime scene image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a bloodstain pattern for analysis"
    )
    
    # Only show the image if something was uploaded
    if uploaded_file is not None:
        # PIL.Image.open reads the uploaded file
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

with col2:
    st.header("🔍 Analysis Results")
    
    # Only run analysis if an image was uploaded
    if uploaded_file is not None:
        
        # Show a spinner while the model is running
        with st.spinner("Analyzing bloodstain patterns..."):
            
            # Convert PIL image to numpy array (what OpenCV and YOLO expect)
            # PIL uses RGB, OpenCV uses BGR — so we convert
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # ── RUN THE MODEL ───────────────────────────────────────
            # This is where your trained YOLOv8 model does its work
            # conf= sets minimum confidence to report a detection
            results = model.predict(img_bgr, conf=confidence, verbose=False)
            
            # results[0] = first image results (we only uploaded one)
            result = results[0]
            
            # ── DRAW BOXES ON IMAGE ─────────────────────────────────
            # result.plot() returns the image with boxes drawn on it
            # It automatically uses the class colors and labels
            annotated = result.plot()
            
            # Convert back from BGR to RGB for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Display the analyzed image
            st.image(annotated_rgb, caption="Detected Patterns", use_column_width=True)
        
        # ── DETECTIONS SUMMARY ──────────────────────────────────────
        # result.boxes contains all detected bounding boxes
        boxes = result.boxes
        
        if len(boxes) == 0:
            st.warning("⚠️ No bloodstain patterns detected. Try lowering the confidence threshold.")
        else:
            st.success(f"✅ Found {len(boxes)} bloodstain pattern(s)")
            
            # ── FORENSIC REPORT ─────────────────────────────────────
            st.markdown("---")
            st.header("📄 Forensic Analysis Report")
            
            # Timestamp
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**Analysis Date:** {now}")
            st.markdown(f"**Image:** {uploaded_file.name}")
            st.markdown(f"**Model:** YOLOv8n — BSARA v1")
            st.markdown("---")
            
            # Loop through each detection and show details
            st.subheader("Detected Patterns:")
            
            # Track which classes were found
            found_classes = []
            
            for i, box in enumerate(boxes):
                # Get class ID and convert to class name
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Get confidence score (0 to 1)
                conf_score = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                found_classes.append(class_name)
                
                # Get class info
                info = CLASS_INFO.get(class_name, {
                    "description": "Unknown pattern type",
                    "severity": "Unknown"
                })
                
                # Display each detection as an expandable section
                with st.expander(f"Detection {i+1}: {class_name.upper()} ({conf_score:.1%} confidence)"):
                    st.markdown(f"**Pattern Type:** {class_name}")
                    st.markdown(f"**Confidence:** {conf_score:.1%}")
                    st.markdown(f"**Location:** x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}")
                    st.markdown(f"**Assessment:** {info['description']}")
                    st.markdown(f"**Event Type:** {info['severity']}")
            
            # ── OVERALL SCENE ASSESSMENT ────────────────────────────
            st.markdown("---")
            st.subheader("🔎 Scene Assessment")
            
            unique_classes = list(set(found_classes))
            
            if "gunshot" in unique_classes:
                st.error("🔴 HIGH PRIORITY: Gunshot backspatter detected — possible firearm involvement")
            if "impact_beating" in unique_classes:
                st.warning("🟠 Blunt force impact spatter detected — possible beating or striking event")
            if "drip" in unique_classes:
                st.info("🔵 Passive drip patterns detected — victim was stationary at this location")
            
            st.markdown("---")
            st.caption("⚠️ DISCLAIMER: This analysis is AI-generated and should not be used as sole forensic evidence. Results must be verified by a certified forensic analyst.")
    
    else:
        # Show placeholder when no image is uploaded
        st.info("👆 Upload an image on the left to begin analysis")