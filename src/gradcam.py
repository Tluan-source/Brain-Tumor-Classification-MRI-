import numpy as np
import cv2
import tensorflow as tf

def make_gradcam_heatmap(model, img_array, last_conv_layer="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_output)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))

    cam = np.zeros(conv_output.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_output[0,:,:,i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam /= cam.max()
    return cam
