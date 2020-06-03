import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from keras.layers import BatchNormalization, Dropout, concatenate, Activation
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from tools.modified_sgd import Modified_SGD
from networks.wide_resnet_builder import create_wide_residual_network


class WideNetwork():
    def __init__(self, input_shape, optimizer, left_classif_factor, right_classif_factor):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def build_net(self, num_outputs):
        embedding_net = create_wide_residual_network(self.input_shape, N=2, k=2,
                                                     dropout=0.6)
        encoded_l = embedding_net(self.left_input)
        encoded_r = embedding_net(self.right_input)

        common_branch = concatenate([encoded_l, encoded_r])

        common_branch = Conv2D(1024, (1, 1), padding="same", activation='relu',
                               kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-3))(common_branch)
        common_branch = Conv2D(1024, (3, 3), padding="same", activation='relu',
                               kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-3))(common_branch)
        common_branch = BatchNormalization()(common_branch)
        common_branch = Flatten()(common_branch)
        common_branch = Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                              kernel_initializer="he_normal")(common_branch)

        siamese_prediction = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(common_branch)

        right_branch = Flatten()(encoded_r)
        right_branch = Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                             kernel_initializer="he_normal")(right_branch)
        right_branch_classif = Dense(num_outputs, activation='softmax',
                                     name="Right_branch_classification")(right_branch)

        left_branch = Flatten()(encoded_l)
        left_branch = Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                            kernel_initializer="he_normal")(left_branch)
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
                            loss_weights={"Left_branch_classification": 0.7,
                                          "Siamese_classification": 0.,
                                          "Right_branch_classification": 0.7})
        return siamese_net
