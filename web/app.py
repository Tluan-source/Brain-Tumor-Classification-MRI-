import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import json
import sys
import os

# sys.path.append("../src")

from src.utils import load_image
from src.gradcam import make_gradcam_heatmap

MODEL_PATH = "/models/brain_tumor_4class.h5"
model = load_model(MODEL_PATH)

with open("/src/labels.json", "r") as f:
    labels = json.load(f)

idx_to_class = {v: k for k, v in labels.items()}

#           Web UI
st.title("Brain Tumor Classification (MRI) Demo")
st.write("Upload một ảnh MRI để dự đoán loại u và hiển thị GradCAM.")

uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh MRI đã upload", use_column_width=True)

    img_array = load_image(uploaded_file)

    # Dự đoán
    preds = model.predict(img_array)[0]
    cls_idx = preds.argmax()
    label = idx_to_class[cls_idx]

    st.subheader(f"🔍 Kết quả dự đoán: **{label}**")

    st.write("Xác suất từng loại u:")
    st.bar_chart(preds)

    # Nếu có u -> tạo heatmap GradCAM
    if label != "no_tumor":
        heatmap = make_gradcam_heatmap(model, img_array)

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

        st.subheader("🔥 Grad-CAM (vùng nghi là khối u)")
        st.image(overlay, channels="BGR", use_column_width=True)

    else:
        st.subheader("✔ Không phát hiện khối u")
