import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

def load_image(img_path, size=224):
    img = image.load_img(img_path, target_size=(size, size))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
