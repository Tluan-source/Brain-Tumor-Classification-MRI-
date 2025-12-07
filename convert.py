import tensorflow as tf
import h5py
from keras.saving import load_model

# Load lại model bằng Keras 3 API
model = load_model("models/brain_tumor_4class.h5", compile=False)

# Save thành SavedModel
tf.saved_model.save(model, "models/brain_tumor_saved_model")

print("DONE: Converted to SavedModel")
