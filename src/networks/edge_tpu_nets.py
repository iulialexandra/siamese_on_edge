"""This is modified from the Horizontal Nets.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

from tensorflow.python.keras.layers import Input, Conv2D, Dense, \
    Flatten, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from hardware.Quantizer import apply_quantization
import tensorflow_model_optimization as tfmot


class HorizontalNetworkOnEdge():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def quantize_model(self, model, quantization_type):
        if quantization_type is None:
            return model
        elif quantization_type == "nullhop":
            return apply_quantization(model)
        elif quantization_type == "edgetpu":
            return tfmot.quantization.keras.quantize_model(model)
        else:
            raise ValueError

    def build_horizontal_block(self, input, block_name):
        conv0 = Conv2D(16, (7, 7), padding="same", activation='relu',
                       kernel_initializer="he_normal",
                       name='{}_conv0'.format(block_name),
                       kernel_regularizer=l2(1e-2))(input)

        conv1 = Conv2D(16, (5, 5), padding="same", activation='relu',
                       kernel_initializer="he_normal",
                       name='{}_conv1'.format(block_name),
                       kernel_regularizer=l2(1e-2))(input)

        conv2 = Conv2D(16, (3, 3), padding="same", activation='relu',
                       kernel_initializer="he_normal",
                       name='{}_conv2'.format(block_name),
                       kernel_regularizer=l2(1e-2))(input)

        return concatenate([conv0, conv1, conv2])

    def build_branch(self, quantization):
        net_input = Input(shape=self.input_shape)

        convnet = self.build_horizontal_block(net_input, "block0")
        convnet = self.build_horizontal_block(convnet, "block1")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = self.build_horizontal_block(convnet, "block2")
        convnet = self.build_horizontal_block(convnet, "block3")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = self.build_horizontal_block(convnet, "block4")
        convnet = self.build_horizontal_block(convnet, "block5")
        convnet = self.build_horizontal_block(convnet, "block6")
        convnet = self.build_horizontal_block(convnet, "block7")
        convnet = MaxPooling2D(padding="valid")(convnet)

        branch_model = Model(
            inputs=net_input, outputs=convnet, name="Branch_model")
        return self.quantize_model(branch_model, quantization)

    def build_trunk(self, left, right, quantization):
        common_concat = concatenate([left, right])
        common_input = Input(shape=common_concat.shape[1:])
        common_branch = Conv2D(128, (3, 3), padding="same",
                               activation='relu',
                               kernel_initializer="he_normal",
                               name='trunk_conv1',
                               kernel_regularizer=l2(1e-2))(common_input)
        common_branch = Conv2D(256, (3, 3), padding="same",
                               activation='relu',
                               kernel_initializer="he_normal",
                               name='trunk_conv2',
                               kernel_regularizer=l2(1e-2))(common_branch)
        common_branch = MaxPooling2D(padding="valid")(common_branch)
        common_branch = Flatten()(common_branch)
        common_branch = Dense(
            128, activation="relu", kernel_regularizer=l2(1e-2),
            kernel_initializer="he_normal",
            name='trunk_dense1')(common_branch)
        common_branch = Dropout(0.5)(common_branch)
        self.trunk_model = Model(
            inputs=common_input, outputs=common_branch, name="Trunk_model")
        return self.quantize_model(
            self.trunk_model, quantization)(common_concat)

    def build_side_classifier(self, num_outputs, input_layer, name):
        side_classif = Flatten()(input_layer)
        side_classif = Dense(
            512, activation="relu", kernel_regularizer=l2(1e-2),
            kernel_initializer="he_normal",
            name='{}_dense0'.format(name))(side_classif)
        side_classif = Dense(
            num_outputs, activation='softmax',
            name="{}_branch_classification".format(name))(
            side_classif)
        return side_classif

    def build_net(self, num_outputs, quantization):

        embedding_model = self.build_branch(quantization)
        encoded_l = embedding_model(self.left_input)
        encoded_r = embedding_model(self.right_input)

        trunk = self.build_trunk(encoded_l, encoded_r, quantization)
        siamese_classifier = Dense(1, activation='sigmoid',
                                   name="Siamese_classification")(trunk)

        left_classifier = self.build_side_classifier(
            num_outputs, encoded_l, "Left")
        right_classifier = self.build_side_classifier(
            num_outputs, encoded_r, "Right")

        siamese_net_train = Model(
            inputs=[self.left_input, self.right_input],
            outputs=[left_classifier, siamese_classifier, right_classifier])

        siamese_net_test = Model(inputs=[self.left_input, self.right_input],
                                 outputs=[trunk])
        return siamese_net_train, siamese_net_test

    def build_edge_net(self, quantization):

        self.embedding_model = self.build_branch(quantization)
        self.encoded_l = self.embedding_model(self.left_input)
        self.encoded_r = self.embedding_model(self.right_input)

        self.trunk = self.build_trunk(
            self.encoded_l, self.encoded_r, quantization)
        #  siamese_classifier = Dense(1, activation='sigmoid',
        #                             name="Siamese_classification")(trunk)

        siamese_net_test = Model(inputs=[self.left_input, self.right_input],
                                 outputs=[self.trunk])
        return siamese_net_test
