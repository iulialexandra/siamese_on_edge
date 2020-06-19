from keras.layers import Input, Lambda, Conv2D, Dense, Flatten, MaxPooling2D, SeparableConv2D
from keras.layers import BatchNormalization, Dropout, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop, Nadam
import tensorflow as tf

class HorizontalNetworkV44():
    def __init__(self, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor

    def horizontal_block(self, input, block_name):
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

    def get_embedding(self, inputs, branch):

        convnet = self.horizontal_block(inputs, "block0" + branch)
        convnet = self.horizontal_block(convnet, "block1" + branch)
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = self.horizontal_block(convnet, "block2" + branch)
        convnet = self.horizontal_block(convnet, "block3" + branch)
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = self.horizontal_block(convnet, "block4" + branch)
        convnet = self.horizontal_block(convnet, "block5" + branch)
        convnet = self.horizontal_block(convnet, "block6" + branch)
        convnet = self.horizontal_block(convnet, "block7" + branch)
        convnet = MaxPooling2D(padding="valid")(convnet)

        return convnet

    def build_siamese(self, num_outputs, model_l, model_r):

        common_branch = concatenate([model_l, model_r])

        common_branch = Conv2D(128, (3, 3), padding="same", activation='relu',
                               kernel_initializer="he_normal", name='center_conv1',
                               kernel_regularizer=l2(1e-2))(common_branch)

        common_branch = Conv2D(256, (3, 3), padding="same", activation='relu',
                               kernel_initializer="he_normal", name='center_conv2',
                               kernel_regularizer=l2(1e-2))(common_branch)
        common_branch = MaxPooling2D(padding="valid")(common_branch)

        common_branch = Flatten()(common_branch)
        common_branch = Dense(128, activation="relu", kernel_regularizer=l2(1e-2), kernel_initializer="he_normal")(
            common_branch)
        common_branch = Dropout(0.5)(common_branch)
        siamese_prediction = Dense(1, activation='sigmoid', name="Siamese_classification")(common_branch)

        right_branch = Flatten()(model_r)
        right_branch = Dense(64, activation="relu", kernel_regularizer=l2(1e-2),
                             kernel_initializer="he_normal", name='right_dense0')(right_branch)
        right_branch_classif = Dense(num_outputs, activation='softmax',
                                     name="Right_branch_classification")(right_branch)

        left_branch = Flatten()(model_l)
        left_branch = Dense(64, activation="relu", kernel_regularizer=l2(1e-2),
                            kernel_initializer="he_normal", name='left_dense0')(left_branch)
        left_branch_classif = Dense(num_outputs, activation='softmax',
                                    name="Left_branch_classification")(left_branch)

        return siamese_prediction, left_branch_classif, right_branch_classif
