from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Input, Dropout, \
    concatenate, Lambda, Subtract
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from tools.modified_sgd import Modified_SGD


class OriginalNetworkV2:
    def __init__(self, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor

    def get_embedding(self, inputs, branch):
        convnet = Conv2D(filters=64, kernel_size=(7, 7),
                         activation='relu',
                         padding="same",
                         kernel_regularizer=l2(1e-3),
                         name='Conv1' + branch)(inputs)
        convnet = MaxPooling2D(padding="same")(convnet)
        convnet = Dropout(0.5)(convnet)

        convnet = Conv2D(filters=128, kernel_size=(5, 5),
                         activation='relu',
                         padding="same",
                         kernel_regularizer=l2(1e-3),
                         name='Conv2' + branch)(convnet)
        convnet = MaxPooling2D(padding="same")(convnet)
        convnet = Dropout(0.5)(convnet)

        convnet = Conv2D(filters=128, kernel_size=(3, 3),
                         activation='relu',
                         padding="same",
                         kernel_regularizer=l2(1e-3),
                         name='Conv3' + branch)(convnet)
        convnet = MaxPool2D(padding="same")(convnet)
        convnet = Dropout(0.5)(convnet)

        convnet = Conv2D(filters=256, kernel_size=(3, 3),
                         activation='relu',
                         padding="same",
                         kernel_regularizer=l2(1e-3),
                         name='Conv4' + branch)(convnet)
        convnet = MaxPool2D(padding="same")(convnet)
        convnet = Dropout(0.5)(convnet)
        convnet = Flatten()(convnet)
        convnet = Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                        kernel_initializer="he_normal", name="Dense1" + branch)(convnet)
        return convnet

    def build_siamese(self, num_outputs, model_l, model_r):

        subtraction_layer = Subtract()([model_l, model_r])
        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(subtraction_layer)

        right_branch_classif = Dense(num_outputs, activation='softmax',
                                     name="Right_branch_classification")(model_r)

        left_branch_classif = Dense(num_outputs, activation='softmax',
                                    name="Left_branch_classification")(model_l)

        return siamese_prediction, left_branch_classif, right_branch_classif
