import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
from gradcam import make_gradcam_heatmap
from utils import load_image

model = load_model("../models/brain_tumor_4class.h5")

with open("labels.json", "r") as f:
    labels = json.load(f)

idx_to_class = {v: k for k, v in labels.items()}

def predict(img_path):
    img_array = load_image(img_path)
    preds = model.predict(img_array)[0]
    cls_idx = preds.argmax()
    label = idx_to_class[cls_idx]

    print("Predicted:", label)

    # Nếu là tumor -> hiện GradCAM
    if label != "no_tumor":
        heatmap = make_gradcam_heatmap(model, img_array)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))

        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        cv2.imshow("GradCAM", overlay)
        cv2.waitKey(0)
    else:
        img = cv2.imread(img_path)
        cv2.imshow("Result", img)
        cv2.waitKey(0)

predict("../data/Testing/glioma_tumor/image.jpg")
