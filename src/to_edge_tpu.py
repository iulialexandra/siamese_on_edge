"""Convert model to edge TPU.

1. Support Keras model.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse

import tensorflow as tf
import numpy as np

#  from tensorflow.python.keras.
from networks.edge_tpu_nets import HorizontalNetworkOnEdge

# force to use CPU
tf.config.set_visible_devices([], 'GPU')

parser = argparse.ArgumentParser()

parser.add_argument("--keras_model", type=str, default="")
parser.add_argument("--saved_model", type=str, default="")
parser.add_argument("--out_tflite", type=str)

args = parser.parse_args()

input_shape = (64, 64, 3)

if args.keras_model != "":
    net = HorizontalNetworkOnEdge(input_shape)

    model = net.build_edge_net(None)

    # load weights
    #  model.load_weights(args.keras_model)
    print("[MESSAGE] Model is loaded")
    model.summary()

    print("[MESSAGE] Get tflite converter")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
elif args.saved_model != "":
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]


def representative_dataset_gen():
    num_calibration_steps = 10
    for _ in range(num_calibration_steps):
        # Get sample input data as a numpy array in a method of your choosing.
        image = tf.random.normal([1, 64, 64, 3])
        #  image = np.random.randint(
        #      -127, 128, size=(1, 64, 64, 3), dtype=np.int8)
        yield [image, image]


converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

print("[MESSAGE] Converting file")
tflite_model = converter.convert()

with tf.io.gfile.GFile(args.out_tflite, "wb") as f:
    f.write(tflite_model)
    print("[MESSAGE] Model write to {}".format(args.out_tflite))
