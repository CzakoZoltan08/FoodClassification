import tensorflow as tf

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from keras import backend as K

from os import path

from . import food_classification_constants as fcc


class FoodClassificationModel:
    def __init__(self, static_files_path):
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        self.RestNet50 = None
        self.root_path = static_files_path
        print("FoodClassification INIT!")

    def create_model(self):
        with self.graph.as_default():
            self.RestNet50 = Sequential()

            weights_file = path.join(
                self.root_path,
                'static',
                'resnet50_weights_notop.h5')

            self.RestNet50.add(
                    ResNet50(
                           include_top=False,
                           pooling=fcc.RESNET50_POOLING_AVERAGE,
                           weights=weights_file))

            self.RestNet50.add(
                    Dense(
                        units=fcc.NUM_CLASSES,
                        activation=fcc.DENSE_LAYER_ACTIVATION))

            self.RestNet50.layers[0].trainable = False

            self.RestNet50.summary()

            weights_path = path.join(
                self.root_path,
                'static',
                'best.hdf5')

            set_session(self.session)
            self.RestNet50.load_weights(weights_path)

    def predict_class(self, image):
        with self.graph.as_default():
            set_session(self.session)

            y_pred = self.RestNet50.predict_classes([[image]])
            confidence = self.RestNet50.predict([[image]])

            confidence_str = "{0:.3f}".format(confidence[0][y_pred][0])

            print("Confidence:")
            print(confidence_str)

            print("Class:")
            if y_pred == 0:
                print("HOT DOG")
                return ("HOT DOG", confidence_str)
            else:
                print("PIZZA")
                return ("PIZZA", confidence_str)
