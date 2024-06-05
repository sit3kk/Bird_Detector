import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from absl import logging

logging.set_verbosity(logging.ERROR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def prediction_MobileNetV2(image_path):
    model_load_path = "saved_models/MobileNetV2/bird_detection_model.h5"
    model = tf.keras.models.load_model(model_load_path)

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return "bird" if predicted_class == 0 else "nonbird"
