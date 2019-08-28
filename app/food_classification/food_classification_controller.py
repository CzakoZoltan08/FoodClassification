from flask import Blueprint, request, jsonify

from keras.preprocessing.image import load_img

import numpy as np

from .food_classification_model import FoodClassificationModel
from .food_classification_service import save_file


food_classification_module = Blueprint(
        'food_classification',
        __name__,
        url_prefix='/',
        static_folder='../static')

food_classification_model = FoodClassificationModel(
        static_files_path=food_classification_module.root_path)
food_classification_model.create_model()
food_classification_model.RestNet50.summary()


@food_classification_module.route('/load-model', methods=['GET'])
def load_model():
    global food_classification_model

    if food_classification_model.RestNet50 is not None:
        return "Model was already loaded!"

    food_classification_model.create_model()

    return "Model was loaded successfully!"


@food_classification_module.route('/predict', methods=['POST'])
def predict():
    global food_classification_model

    # check if the post request has the file part
    if 'file' not in request.files:
        return 'No file found'

    user_file = request.files['file']

    file_path = save_file(user_file, food_classification_module.root_path)

    image = np.asarray(load_img(file_path, target_size=(224, 224)))

    (class_name, confidence) = food_classification_model.predict_class(image)

    return jsonify(className=class_name, confidence=confidence)
