import numpy as np
import cv2
import tensorflow as tf

def make_gradcam_heatmap(model, img_array, orig_image, last_conv_layer=None):

    if last_conv_layer is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer.name
                break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]

    heatmap = tf.zeros(shape=conv_outputs.shape[0:2])
    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig_image, 0.6, heatmap_color, 0.4, 0)

    return overlay
