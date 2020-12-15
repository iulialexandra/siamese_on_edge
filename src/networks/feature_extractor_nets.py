from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Input, Dropout, \
    concatenate, Lambda
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from tools.modified_sgd import Modified_SGD


class ResnetFeaturesNetV1:
    def __init__(self, input_shape, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def build_net(self, num_outputs):
        convnet = Sequential()
        convnet.add(Flatten())
        convnet.add(Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal"))
        convnet.add(Dense(1024, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal"))
        convnet.add(Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal"))

        encoded_l = convnet(self.left_input)
        encoded_r = convnet(self.right_input)

        DistanceLayer = Lambda(lambda tensors: tf.square(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        dist = DistanceLayer([encoded_l, encoded_r])
        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(dist)

        right_branch_classif = Dense(num_outputs, activation='softmax',
                                     name="Right_branch_classification")(encoded_r)

        left_branch_classif = Dense(num_outputs, activation='softmax',
                                    name="Left_branch_classification")(encoded_l)

        siamese_net = Model(inputs=[self.left_input, self.right_input],
                            outputs=[left_branch_classif, siamese_prediction, right_branch_classif])

        siamese_net.compile(loss={"Left_branch_classification": "categorical_crossentropy",
                                  "Siamese_classification": "binary_crossentropy",
                                  "Right_branch_classification": "categorical_crossentropy"},
                            optimizer=self.optimizer,
                            metrics={"Left_branch_classification": "accuracy",
                                     "Siamese_classification": "accuracy",
                                     "Right_branch_classification": "accuracy"},
                            loss_weights={"Left_branch_classification": self.left_classif_factor,
                                          "Siamese_classification": self.siamese_factor,
                                          "Right_branch_classification": self.right_classif_factor})
        return siamese_net


class ResnetFeaturesNetV2:
    def __init__(self, input_shape, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def build_net(self, num_outputs):
        convnet = Sequential()
        convnet.add(Conv2D(filters=64, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           input_shape=self.input_shape,
                           kernel_regularizer=l2(1e-3),
                           name='Conv1'))
        convnet.add(Flatten())
        convnet.add(Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal"))
        convnet.add(Dense(1024, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal"))
        convnet.add(Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal"))

        encoded_l = convnet(self.left_input)
        encoded_r = convnet(self.right_input)

        DistanceLayer = Lambda(lambda tensors: tf.square(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        dist = DistanceLayer([encoded_l, encoded_r])
        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(dist)

        right_branch_classif = Dense(num_outputs, activation='softmax',
                                     name="Right_branch_classification")(encoded_r)

        left_branch_classif = Dense(num_outputs, activation='softmax',
                                    name="Left_branch_classification")(encoded_l)

        siamese_net = Model(inputs=[self.left_input, self.right_input],
                            outputs=[left_branch_classif, siamese_prediction, right_branch_classif])

        siamese_net.compile(loss={"Left_branch_classification": "categorical_crossentropy",
                                  "Siamese_classification": "binary_crossentropy",
                                  "Right_branch_classification": "categorical_crossentropy"},
                            optimizer=self.optimizer,
                            metrics={"Left_branch_classification": "accuracy",
                                     "Siamese_classification": "accuracy",
                                     "Right_branch_classification": "accuracy"},
                            loss_weights={"Left_branch_classification": self.left_classif_factor,
                                          "Siamese_classification": self.siamese_factor,
                                          "Right_branch_classification": self.right_classif_factor})
        return siamese_net

class SimclrNetV1:
    def __init__(self, input_shape, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def build_net(self, num_outputs):
        net_input = Input(shape=self.input_shape)
        net = Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal")(net_input)
        net = Dense(1024, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal")(net)
        net = Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal")(net)

        embedding = Model(inputs=net_input, outputs=net)

        encoded_l = embedding(self.left_input)
        encoded_r = embedding(self.right_input)

        DistanceLayer = Lambda(lambda tensors: tf.square(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        dist = DistanceLayer([encoded_l, encoded_r])
        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(dist)

        right_branch_classif = Dense(num_outputs, activation='softmax',
                                     name="Right_branch_classification")(encoded_r)

        left_branch_classif = Dense(num_outputs, activation='softmax',
                                    name="Left_branch_classification")(encoded_l)

        siamese_net = Model(inputs=[self.left_input, self.right_input],
                            outputs=[left_branch_classif, siamese_prediction, right_branch_classif])

        siamese_net.compile(loss={"Left_branch_classification": "categorical_crossentropy",
                                  "Siamese_classification": "binary_crossentropy",
                                  "Right_branch_classification": "categorical_crossentropy"},
                            optimizer=self.optimizer,
                            metrics={"Left_branch_classification": "accuracy",
                                     "Siamese_classification": "accuracy",
                                     "Right_branch_classification": "accuracy"},
                            loss_weights={"Left_branch_classification": self.left_classif_factor,
                                          "Siamese_classification": self.siamese_factor,
                                          "Right_branch_classification": self.right_classif_factor})
        return siamese_net, embedding