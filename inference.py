import os.path

import cv2
import numpy as np
import tensorflow as tf

from data_model import ImageModel, PredictionClass

MODEL_PATH_DIR = "models"
IMAGE_DATA_DIR = "image_data"


def load_model(model: str):
    model_path = os.path.join(MODEL_PATH_DIR, model)
    return tf.keras.models.load_model(model_path)


def prepare_image(image_dir: str):
    size = 180, 180
    temp_img = cv2.imread(os.path.join(IMAGE_DATA_DIR, image_dir))
    temp_img = cv2.resize(temp_img, size)
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    temp_img = temp_img.astype('float32') / 255.0
    return np.expand_dims(temp_img, axis=0)


def post_process(pred: float, threshold: float = 0.5):
    if pred > threshold:
        return PredictionClass.pneumonia.value
    else:
        return PredictionClass.normal.value


def predict_image_model(model: ImageModel, image_path: str):
    model = load_model(model.value)
    image = prepare_image(image_path)
    pred = model.predict(image)
    return post_process(pred)
