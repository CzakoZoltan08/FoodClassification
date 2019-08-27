from flask import Blueprint, request
from werkzeug.utils import secure_filename

from keras.models import model_from_json
from keras.preprocessing.image import load_img

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from os import path

import numpy as np

from flask import current_app as app


# Fixed for Hot Dog & Pizza color images
CHANNELS = 3

IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# The name of the .h5 file containing the pretrained weights for the ResNet50
WEIGHTS_FILE = 'resnet50_weights_notop.h5'


model = None
graph = tf.get_default_graph()
sess = tf.Session()


def load_model_from_json_and_h5(json_file_name, h5_file_name):
    json_file = open(json_file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(h5_file_name)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    # Define X_test & Y_test data first
    loaded_model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    return loaded_model


def predict_class(image):
    global model
    global graph
    global session

    with graph.as_default():
        set_session(sess)
        y_pred = model.predict_classes([[image]])
        confidence = model.predict([[image]])
        print("Confidence:")
        print("{0:.2f}".format(confidence[0][y_pred][0]))

        print("Class:")
        if y_pred == 0:
            print("HOT DOG")
        else:
            print("PIZZA")


# Define the blueprint: 'auth', set its url prefix: app.url/auth
food_classification_module = Blueprint(
        'food_classification',
        __name__,
        url_prefix='/',
        static_folder='../static')

filepath_h5 = path.join(
        food_classification_module.root_path,
        'static',
        'pizza_vs_hotdog.h5')
filepath_json = path.join(
        food_classification_module.root_path,
        'static',
        'pizza_vs_hotdog.json')


@food_classification_module.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'


@food_classification_module.route('/load-model', methods=['GET'])
def load_model():
    global model
    global graph
    global sess

    if model is not None:
        return "Model was already loaded!"

    with graph.as_default():

        model = Sequential()

        weights_file = path.join(
            food_classification_module.root_path,
            'static',
            'resnet50_weights_notop.h5')

        model.add(
                ResNet50(
                        include_top=False,
                        pooling=RESNET50_POOLING_AVERAGE,
                        weights=weights_file))

        model.add(
                Dense(
                        app.config["NUM_CLASSES"],
                        activation=DENSE_LAYER_ACTIVATION))

        # Say not to train first layer (ResNet) model as it is already trained
        model.layers[0].trainable = False

        model.summary()

        weights_path = path.join(
            food_classification_module.root_path,
            'static',
            'best.hdf5')

        set_session(sess)
        model.load_weights(weights_path)

    return "Model was loaded successfully!"


@food_classification_module.route('/predict', methods=['POST'])
def predict():
    # check if the post request has the file part
    if 'file' not in request.files:
        return 'No file found'

    user_file = request.files['file']

    filename = secure_filename(user_file.filename)
    filepath = path.join(
            food_classification_module.root_path,
            'static',
            filename)
    user_file.save(filepath)

    image = np.asarray(load_img(filepath, target_size=(224, 224)))

    predict_class(image)

    return "Done!"
