import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_image
from src.gradcam import make_gradcam_heatmap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "brain_tumor_4class.h5"))
LABEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "src", "labels.json"))

model = load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    labels = json.load(f)

idx_to_class = {v: k for k, v in labels.items()}

# UI
st.title("Brain Tumor Classification (MRI) Demo")
st.write("Upload một ảnh MRI để dự đoán loại u và hiển thị GradCAM.")

uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Ảnh MRI đã upload", use_column_width=True)

    img_array = load_image(uploaded_file)

    preds = model.predict(img_array)[0]
    cls_idx = preds.argmax()
    label = idx_to_class[cls_idx]

    st.subheader(f"🔍 Kết quả dự đoán: **{label}**")

    st.write("Xác suất từng loại u:")
    st.bar_chart(preds)

    if label != "no_tumor":
        orig = np.array(img)
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

        heatmap_img = make_gradcam_heatmap(model, img_array, orig_bgr)

        st.subheader("Grad-CAM (vùng nghi là khối u)")
        st.image(heatmap_img, channels="BGR", use_column_width=True)

    else:
        st.subheader("Không phát hiện khối u")
