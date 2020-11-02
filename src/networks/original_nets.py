from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Input, Dropout, \
    concatenate, Lambda
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from tools.modified_sgd import Modified_SGD


class OriginalNetworkV2:
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
        convnet.add(Conv2D(filters=64, kernel_size=(7, 7),
                           activation='relu',
                           padding="same",
                           input_shape=self.input_shape,
                           kernel_regularizer=l2(1e-3),
                           name='Conv1'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=128, kernel_size=(5, 5),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv2'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=128, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv3'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=256, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv4'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))
        convnet.add(Flatten())
        convnet.add(Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal"))

        encoded_l = convnet(self.left_input)
        encoded_r = convnet(self.right_input)

        DistanceLayer = Lambda(lambda tensors: tf.square(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        common_branch = DistanceLayer([encoded_l, encoded_r])
        common_branch = Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                              kernel_initializer="he_normal")(common_branch)
        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(common_branch)

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


class OriginalNetworkV3:
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
        convnet.add(Conv2D(filters=64, kernel_size=(7, 7),
                           activation='relu',
                           padding="same",
                           input_shape=self.input_shape,
                           kernel_regularizer=l2(1e-3),
                           name='Conv1'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=128, kernel_size=(5, 5),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv2'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=128, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv3'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=256, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv4'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))
        convnet.add(Flatten())
        convnet.add(Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                          kernel_initializer="he_normal"))

        encoded_l = convnet(self.left_input)
        encoded_r = convnet(self.right_input)

        common_branch = concatenate([encoded_l, encoded_r])
        common_branch = Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                              kernel_initializer="he_normal")(common_branch)

        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(common_branch)

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


class OriginalNetworkV4:
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
        convnet.add(Conv2D(filters=64, kernel_size=(7, 7),
                           activation='relu',
                           padding="same",
                           input_shape=self.input_shape,
                           kernel_regularizer=l2(1e-3),
                           name='Conv1'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=128, kernel_size=(5, 5),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv2'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=128, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv3'))
        convnet.add(MaxPool2D(padding="same"))
        convnet.add(Dropout(0.5))

        convnet.add(Conv2D(filters=256, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-3),
                           name='Conv4'))
        convnet.add(MaxPool2D(padding="same"))

        encoded_l = convnet(self.left_input)
        encoded_r = convnet(self.right_input)

        common_branch = concatenate([encoded_l, encoded_r])
        common_branch = Conv2D(512, (3, 3), padding="same", activation='relu',
                               kernel_initializer="he_normal", name='center_conv1',
                               kernel_regularizer=l2(1e-2))(common_branch)
        common_branch = Flatten()(common_branch)
        common_branch = Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                              kernel_initializer="he_normal")(common_branch)
        common_branch = Dropout(0.5)(common_branch)

        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(common_branch)

        right_branch_classif = Flatten()(encoded_r)
        right_branch_classif = Dense(num_outputs, activation='softmax',
                                     name="Right_branch_classification")(right_branch_classif)

        left_branch_classif = Flatten()(encoded_l)
        left_branch_classif = Dense(num_outputs, activation='softmax',
                                    name="Left_branch_classification")(left_branch_classif)

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


class OriginalNetworkV333:
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
        convnet.add(Conv2D(filters=32, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           input_shape=self.input_shape,
                           kernel_regularizer=l2(1e-2),
                           name='Conv1'))
        convnet.add(BatchNormalization())

        convnet.add(Conv2D(filters=32, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-2),
                           name='Conv2'))
        convnet.add(BatchNormalization())

        convnet.add(Conv2D(filters=32, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-2),
                           name='Conv3'))
        convnet.add(BatchNormalization())

        convnet.add(Conv2D(filters=32, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-2),
                           name='Conv4'))
        convnet.add(BatchNormalization())

        convnet.add(Conv2D(filters=32, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-2),
                           name='Conv5'))
        convnet.add(BatchNormalization())
        convnet.add(MaxPool2D(padding="same"))

        convnet.add(Conv2D(filters=32, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-2),
                           name='Conv6'))
        convnet.add(BatchNormalization())
        convnet.add(MaxPool2D(padding="same"))

        convnet.add(Conv2D(filters=32, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-2),
                           name='Conv7'))
        convnet.add(BatchNormalization())
        convnet.add(MaxPool2D(padding="same"))

        convnet.add(Conv2D(filters=32, kernel_size=(3, 3),
                           activation='relu',
                           padding="same",
                           kernel_regularizer=l2(1e-2),
                           name='Conv8'))
        convnet.add(BatchNormalization())
        convnet.add(MaxPool2D(padding="same"))

        encoded_l = convnet(self.left_input)
        encoded_r = convnet(self.right_input)

        common_branch = concatenate([encoded_l, encoded_r])
        # common_branch = Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
        #                       kernel_initializer="he_normal")(common_branch)

        common_branch = Conv2D(512, (3, 3), padding="same", activation='relu',
                               kernel_initializer="he_normal", name='center_conv1',
                               kernel_regularizer=l2(1e-2))(common_branch)
        common_branch = Flatten()(common_branch)
        common_branch = Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                              kernel_initializer="he_normal")(common_branch)
        common_branch = Dropout(0.5)(common_branch)

        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(common_branch)

        right_branch = Flatten()(encoded_r)
        right_branch = Dense(512, activation="relu", kernel_regularizer=l2(1e-2),
                             kernel_initializer="he_normal", name='right_dense0')(right_branch)
        right_branch_classif = Dense(num_outputs, activation='softmax',
                                     name="Right_branch_classification")(right_branch)

        left_branch = Flatten()(encoded_l)
        left_branch = Dense(512, activation="relu", kernel_regularizer=l2(1e-2),
                            kernel_initializer="he_normal", name='left_dense0')(left_branch)
        left_branch_classif = Dense(num_outputs, activation='softmax',
                                    name="Left_branch_classification")(left_branch)

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
