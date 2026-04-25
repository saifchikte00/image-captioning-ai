import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import tempfile
from io import BytesIO
from datetime import datetime

# ----------------------------
# Page Settings
# ----------------------------
st.set_page_config(
    page_title="Vision AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Session State for History
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Neon AI Theme CSS
# ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(0,229,255,0.15), transparent 25%),
                    radial-gradient(circle at top right, rgba(139,92,246,0.18), transparent 25%),
                    radial-gradient(circle at bottom left, rgba(255,78,205,0.12), transparent 25%),
                    linear-gradient(135deg, #050816, #0a0f1f, #111827);
        color: white;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: white !important;
    }

    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 31, 0.95) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.15);
    }

    .card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(0, 229, 255, 0.18);
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.08);
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }

    .caption-box {
        background: rgba(0, 229, 255, 0.10);
        border: 1px solid rgba(0, 229, 255, 0.35);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.10);
        padding: 18px;
        border-radius: 14px;
        font-size: 22px;
        font-weight: 600;
        color: #a5f3fc !important;
        margin-top: 15px;
    }

    .object-box {
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.35);
        box-shadow: 0 0 18px rgba(139, 92, 246, 0.08);
        padding: 16px;
        border-radius: 14px;
        font-size: 18px;
        color: #ddd6fe !important;
        margin-top: 15px;
    }

    .summary-box {
        background: rgba(255, 78, 205, 0.10);
        border: 1px solid rgba(255, 78, 205, 0.30);
        box-shadow: 0 0 18px rgba(255, 78, 205, 0.08);
        padding: 16px;
        border-radius: 14px;
        font-size: 18px;
        color: #fbcfe8 !important;
        margin-top: 15px;
    }

    .hazard-box {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.35);
        box-shadow: 0 0 18px rgba(239, 68, 68, 0.08);
        padding: 16px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: 600;
        color: #fecaca !important;
        margin-top: 15px;
    }

    .audio-box {
        background: rgba(34, 197, 94, 0.14);
        border: 1px solid rgba(34, 197, 94, 0.35);
        box-shadow: 0 0 18px rgba(34, 197, 94, 0.08);
        padding: 16px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: 600;
        color: #bbf7d0 !important;
        margin-top: 15px;
    }

    .footer-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 229, 255, 0.15);
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.05);
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(0,229,255,0.12);
    }

    .stButton > button {
        background: linear-gradient(90deg, #00e5ff, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(0,229,255,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("Configure your AI captioning pipeline")

show_boxes = st.sidebar.checkbox("Show Object Detection Output", value=True)
show_summary = st.sidebar.checkbox("Show Smart Summary", value=True)
show_audio = st.sidebar.checkbox("Enable Audio Narration", value=True)
show_hazard = st.sidebar.checkbox("Enable Hazard Awareness", value=True)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Features")
st.sidebar.write("✅ BLIP Captioning")
st.sidebar.write("✅ YOLOv8 Object Detection")
st.sidebar.write("✅ Real-time Image Analysis")
st.sidebar.write("✅ Audio Caption Narration")
st.sidebar.write("✅ Risk / Hazard Awareness")
st.sidebar.write("✅ Report Download")
st.sidebar.write("✅ Result History")
st.sidebar.write("✅ Annotated Image Download")

# ----------------------------
# Header
# ----------------------------
st.title("🧠 Vision AI Captioning Dashboard")
st.markdown("A futuristic AI-powered dashboard for **real-time image captioning**, **object detection**, **hazard awareness**, **audio narration**, and **smart reporting** using **BLIP + YOLOv8**")

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model, device

@st.cache_resource
def load_yolo():
    return YOLO("notebook/yolov8n.pt")

processor, blip_model, device = load_blip()
yolo_model = load_yolo()

# ----------------------------
# Hazard Rules
# ----------------------------
hazard_objects = {
    "knife", "scissors", "car", "bus", "truck", "motorcycle",
    "train", "traffic light", "fire hydrant", "airplane", "boat"
}

# ----------------------------
# Upload Section
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # BLIP Caption Generation
    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)

    # YOLO Object Detection
    with st.spinner("Detecting objects..."):
        results = yolo_model(image_np, conf=confidence_threshold)

        detected_objects = []
        annotated_image = image_np.copy()

        for result in results:
            boxes = result.boxes
            annotated_image = result.plot()

            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = yolo_model.names[cls_id]
                detected_objects.append(label)

    unique_objects = sorted(list(set(detected_objects)))
    object_count = len(unique_objects)

    # Hazard Awareness Logic
    detected_hazards = [obj for obj in unique_objects if obj in hazard_objects]
    hazard_message = None
    if detected_hazards:
        hazard_message = f"⚠️ Potential hazard-related objects detected: {', '.join(detected_hazards)}"

    # Audio Narration Logic
    audio_file_path = None
    if show_audio:
        try:
            tts = gTTS(text=caption, lang="en")
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_audio.name)
            audio_file_path = temp_audio.name
        except Exception as e:
            st.warning(f"Audio generation failed: {e}")

    # Save History
    history_entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "caption": caption,
        "objects": ", ".join(unique_objects) if unique_objects else "None",
        "hazards": ", ".join(detected_hazards) if detected_hazards else "None"
    }

    if not st.session_state.history or st.session_state.history[-1] != history_entry:
        st.session_state.history.append(history_entry)

    # Smart Summary
    if hazard_message:
        summary_text = f"This image likely contains {', '.join(unique_objects)}. Caption: '{caption}'. Hazard note: {', '.join(detected_hazards)} may require attention."
    else:
        summary_text = f"This image likely contains {', '.join(unique_objects)}. The generated caption suggests: '{caption}'."

    # Prepare Report Text
    report_text = f"""
Vision AI Dashboard - Analysis Report
-------------------------------------
Timestamp: {history_entry['time']}

Generated Caption:
{caption}

Detected Objects:
{history_entry['objects']}

Hazard Detection:
{history_entry['hazards']}

Smart Summary:
{summary_text}
"""

    # Prepare Annotated Image Download
    annotated_pil = Image.fromarray(annotated_image)
    img_buffer = BytesIO()
    annotated_pil.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="card"><h3>🧠 Caption</h3><p>Generated</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="card"><h3>📦 Objects</h3><p>{object_count}</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><h3>🤖 Model</h3><p>BLIP + YOLOv8</p></div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card"><h3>⚡ Status</h3><p>Success</p></div>', unsafe_allow_html=True)

    # Main Layout
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("🖼️ Uploaded Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.markdown(
            f'<div class="caption-box">🧠 Generated Caption: {caption}</div>',
            unsafe_allow_html=True
        )

        if unique_objects:
            st.markdown(
                f'<div class="object-box">📦 Detected Objects: {", ".join(unique_objects)}</div>',
                unsafe_allow_html=True
            )

        if show_hazard and hazard_message:
            st.markdown(
                f'<div class="hazard-box">{hazard_message}</div>',
                unsafe_allow_html=True
            )

        if show_audio and audio_file_path:
            st.markdown(
                '<div class="audio-box">🔊 Audio Caption Narration Ready</div>',
                unsafe_allow_html=True
            )
            st.audio(audio_file_path)

        st.download_button(
            label="📄 Download Analysis Report",
            data=report_text,
            file_name="analysis_report.txt",
            mime="text/plain"
        )

    with right_col:
        if show_boxes and unique_objects:
            st.subheader("🎯 Object Detection Output")
            st.image(annotated_image, caption="YOLOv8 Detection", use_container_width=True)

            st.download_button(
                label="🖼️ Download Annotated Detection Image",
                data=img_bytes,
                file_name="annotated_detection.png",
                mime="image/png"
            )

        if show_summary:
            st.markdown(
                f'<div class="summary-box">📝 Smart Summary: {summary_text}</div>',
                unsafe_allow_html=True
            )

    # History Section
    st.markdown("---")
    st.subheader("📜 Result History (Current Session)")

    if st.session_state.history:
        for idx, entry in enumerate(reversed(st.session_state.history[-5:]), start=1):
            st.markdown(
                f"""
                <div class="card">
                    <b>#{idx}</b><br>
                    ⏰ {entry['time']}<br>
                    🧠 <b>Caption:</b> {entry['caption']}<br>
                    📦 <b>Objects:</b> {entry['objects']}<br>
                    ⚠️ <b>Hazards:</b> {entry['hazards']}
                </div>
                """,
                unsafe_allow_html=True
            )

    # Footer
    st.markdown(
        '<div class="footer-box">🚀 Vision AI Dashboard | BLIP + YOLOv8 + Hazard + Audio + Report Export + History | Neon AI Theme</div>',
        unsafe_allow_html=True
    )

else:
    st.info("👆 Please upload an image to begin analysis.")