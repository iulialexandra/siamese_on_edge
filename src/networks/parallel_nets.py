from keras.layers import Input, Lambda, Conv2D, Dense, Flatten, MaxPooling2D, SeparableConv2D
from keras.layers import BatchNormalization, Dropout, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from networks.wide_resnet_builder import create_wide_residual_network
import tensorflow as tf


class ParallelNetworkV0():
    def __init__(self, input_shape, optimizer, left_classif_factor, right_classif_factor, siamese_factor):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.left_classif_factor = left_classif_factor
        self.right_classif_factor = right_classif_factor
        self.siamese_factor = siamese_factor
        self.left_input = Input(self.input_shape, name="Left_input")
        self.right_input = Input(self.input_shape, name="Right_input")

    def build_net(self, num_outputs):
        def parallel_block(next_activ, block_name, kernel_size, depth):
            for layer_idx in range(depth):
                num_kernels = 16 * (2 ** layer_idx)

                if layer_idx != depth - 1:
                    activ = "relu"
                else:
                    activ = None

                next_activ = Conv2D(num_kernels, (kernel_size, kernel_size), padding="same", activation=activ,
                                    kernel_initializer="he_normal", name='{}_conv{}'.format(block_name, layer_idx),
                                    kernel_regularizer=l2(1e-2))(next_activ)
                if layer_idx % 2 == 0:
                    next_activ = MaxPooling2D(padding="valid")(next_activ)

            return next_activ

        def merge_image_proto_branches(input_data, branch, block_name):
            def get_class(embedding, side):
                classif_branch = Flatten()(embedding)
                classif_branch = Dense(512, activation="relu", kernel_regularizer=l2(1e-2),
                                       kernel_initializer="he_normal", name='right_dense_{}_{}'.format(block_name, side))(classif_branch)

                classif_name = "{}_{}_branch_classification".format(block_name, side)
                classif_branch = Dense(num_outputs, activation='softmax', name=classif_name)(classif_branch)
                return classif_branch, classif_name

            embedding = Model(inputs=input_data, outputs=branch)

            encoded_l = embedding(self.left_input)
            encoded_r = embedding(self.right_input)

            common_branch = concatenate([encoded_l, encoded_r])

            next_activ = Conv2D(32, (1, 1), padding="same", activation="relu",
                                kernel_initializer="he_normal", name='{}_conv'.format(block_name),
                                kernel_regularizer=l2(1e-2))(common_branch)

            classif_branch_right, classif_name_right = get_class(encoded_r, "right")
            classif_branch_left, classif_name_left = get_class(encoded_r, "left")

            return next_activ, classif_branch_right, classif_name_right, classif_branch_left, classif_name_left

        net_input = Input(shape=self.input_shape)
        branch3 = parallel_block(next_activ=net_input, block_name="3x3", kernel_size=3, depth=4)
        branch5 = parallel_block(next_activ=net_input, block_name="5x5", kernel_size=5, depth=4)
        branch7 = parallel_block(next_activ=net_input, block_name="7x7", kernel_size=7, depth=4)

        branch3_concat, branch3_classif_right, branch3_classif_name_right, branch3_classif_left, branch3_classif_name_left = merge_image_proto_branches(net_input, branch3, "3x3")
        branch5_concat, branch5_classif_right, branch5_classif_name_right, branch5_classif_left, branch5_classif_name_left = merge_image_proto_branches(net_input, branch5, "5x5")
        branch7_concat, branch7_classif_right, branch7_classif_name_right, branch7_classif_left, branch7_classif_name_left = merge_image_proto_branches(net_input, branch7, "7x7")

        final_branch = concatenate([branch3_concat, branch5_concat, branch7_concat])

        final_branch = Conv2D(128, (3, 3), padding="same", activation='relu',
                              kernel_initializer="he_normal", name='center_conv1',
                              kernel_regularizer=l2(1e-2))(final_branch)
        final_branch = MaxPooling2D(padding="valid")(final_branch)

        final_branch = Conv2D(256, (3, 3), padding="same", activation='relu',
                              kernel_initializer="he_normal", name='center_conv2',
                              kernel_regularizer=l2(1e-2))(final_branch)
        final_branch = MaxPooling2D(padding="valid")(final_branch)

        final_branch = Flatten()(final_branch)
        final_branch = Dropout(0.5)(final_branch)

        siamese_prediction = Dense(1, activation='sigmoid', name="Siamese_classification")(final_branch)

        model_output = [siamese_prediction, branch3_classif_right, branch3_classif_left, branch5_classif_right, branch5_classif_left, branch7_classif_right, branch7_classif_left]

        siamese_net = Model(inputs=[self.left_input, self.right_input],
                            outputs=model_output)

        losses_classif_names = [branch3_classif_name_right, branch5_classif_name_right, branch7_classif_name_right, branch3_classif_name_left, branch5_classif_name_left,
                                branch7_classif_name_left]

        losses_classif_dict = {"Siamese_classification": "binary_crossentropy"}
        metrics = {"Siamese_classification": "accuracy"}
        loss_weights = {"Siamese_classification": self.siamese_factor}


        for name in losses_classif_names:
            losses_classif_dict[name] = "categorical_crossentropy"
            metrics[name] = "accuracy"
            loss_weights[name] = self.left_classif_factor

        siamese_net.compile(loss=losses_classif_dict,
                            optimizer=self.optimizer,
                            metrics=metrics,
                            loss_weights=loss_weights)
        return siamese_net, losses_classif_names
