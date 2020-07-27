from keras.layers import Input, Lambda, Conv2D, Dense, Flatten, MaxPooling2D, SeparableConv2D
from keras.layers import BatchNormalization, Dropout, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from networks.wide_resnet_builder import create_wide_residual_network
import tensorflow as tf


class HorizontalNetworkV5():
    def __init__(self, input_shape, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def build_net(self, num_outputs):
        def horizontal_block(input, block_name):
            conv0 = Conv2D(8, (9, 9), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv0'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv0 = BatchNormalization(name='{}_conv0_batchnorm'.format(block_name))(conv0)

            conv1 = Conv2D(8, (7, 7), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv1'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv1 = BatchNormalization(name='{}_conv1_batchnorm'.format(block_name))(conv1)

            conv2 = Conv2D(8, (5, 5), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv2'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv2 = BatchNormalization(name='{}_conv2_batchnorm'.format(block_name))(conv2)

            conv3 = Conv2D(8, (3, 3), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv3'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv3 = BatchNormalization(name='{}_conv3_batchnorm'.format(block_name))(conv3)

            return concatenate([conv0, conv1, conv2, conv3])

        net_input = Input(shape=self.input_shape)

        convnet = horizontal_block(net_input, "block0")

        convnet = horizontal_block(convnet, "block1")

        convnet = horizontal_block(convnet, "block2")

        convnet = horizontal_block(convnet, "block3")

        convnet = horizontal_block(convnet, "block4")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block5")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block6")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block7")
        convnet = MaxPooling2D(padding="valid")(convnet)

        embedding = Model(inputs=net_input, outputs=convnet)

        encoded_l = embedding(self.left_input)
        encoded_r = embedding(self.right_input)

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


class HorizontalNetworkV44():
    def __init__(self, input_shape, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def build_net(self, num_outputs):
        def horizontal_block(input, block_name):
            conv0 = Conv2D(16, (7, 7), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv0'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)

            conv1 = Conv2D(16, (5, 5), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv1'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)

            conv2 = Conv2D(16, (3, 3), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv2'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)

            return concatenate([conv0, conv1, conv2])

        net_input = Input(shape=self.input_shape)

        convnet = horizontal_block(net_input, "block0")
        convnet = horizontal_block(convnet, "block1")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block2")
        convnet = horizontal_block(convnet, "block3")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block4")
        convnet = horizontal_block(convnet, "block5")
        convnet = horizontal_block(convnet, "block6")
        convnet = horizontal_block(convnet, "block7")

        convnet = MaxPooling2D(padding="valid")(convnet)

        embedding = Model(inputs=net_input, outputs=convnet)

        encoded_l = embedding(self.left_input)
        encoded_r = embedding(self.right_input)

        common_branch = concatenate([encoded_l, encoded_r])

        common_branch = Conv2D(128, (3, 3), padding="same", activation='relu',
                               kernel_initializer="he_normal", name='center_conv1',
                               kernel_regularizer=l2(1e-2))(common_branch)

        common_branch = Conv2D(256, (3, 3), padding="same", activation='relu',
                               kernel_initializer="he_normal", name='center_conv2',
                               kernel_regularizer=l2(1e-2))(common_branch)
        common_branch = MaxPooling2D(padding="valid")(common_branch)

        common_branch = Flatten()(common_branch)
        common_branch = Dense(128, activation="relu", kernel_regularizer=l2(1e-2), kernel_initializer="he_normal")(common_branch)
        common_branch = Dropout(0.5)(common_branch)
        siamese_prediction = Dense(1, activation='sigmoid', name="Siamese_classification")(common_branch)

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


class HorizontalNetworkV333():
    def __init__(self, input_shape, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def build_net(self, num_outputs):
        def horizontal_block(input, block_name):
            conv0 = Conv2D(8, (3, 3), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv0'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv0 = BatchNormalization(name='{}_conv0_batchnorm'.format(block_name))(conv0)

            conv1 = Conv2D(8, (3, 3), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv1'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv1 = BatchNormalization(name='{}_conv1_batchnorm'.format(block_name))(conv1)

            conv2 = Conv2D(8, (3, 3), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv2'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv2 = BatchNormalization(name='{}_conv2_batchnorm'.format(block_name))(conv2)

            conv3 = Conv2D(8, (3, 3), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv3'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv3 = BatchNormalization(name='{}_conv3_batchnorm'.format(block_name))(conv3)

            return concatenate([conv0, conv1, conv2, conv3])

        net_input = Input(shape=self.input_shape)

        convnet = horizontal_block(net_input, "block0")

        convnet = horizontal_block(convnet, "block1")

        convnet = horizontal_block(convnet, "block2")

        convnet = horizontal_block(convnet, "block3")

        convnet = horizontal_block(convnet, "block4")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block5")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block6")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block7")
        convnet = MaxPooling2D(padding="valid")(convnet)

        embedding = Model(inputs=net_input, outputs=convnet)

        encoded_l = embedding(self.left_input)
        encoded_r = embedding(self.right_input)

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
