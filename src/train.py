import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import json

IMG_SIZE = 224
BATCH = 32

train_dir = "../data/Training"
test_dir = "../data/Testing"

train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                               zoom_range=0.2, horizontal_flip=True).flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, class_mode="categorical"
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, class_mode="categorical"
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(4, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)

for layer in base.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_gen, epochs=10, validation_data=test_gen)

# Save model
model.save("../models/brain_tumor_4class.h5")

# Save labels
with open("labels.json", "w") as f:
    json.dump(train_gen.class_indices, f)
