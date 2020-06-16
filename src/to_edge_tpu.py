"""Convert model to edge TPU.

1. Support Keras model.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse

import tensorflow as tf

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

    model = net.build_edge_net("edgetpu")

    # load weights
    model.load_weights(args.keras_model)
    print("[MESSAGE] Model is loaded")
    model.summary()

    print("[MESSAGE] Get tflite converter")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
elif args.saved_model != "":
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)

print("[MESSAGE] Converting file")
tflite_model = converter.convert()

with tf.io.gfile.GFile(args.out_tflite, "wb") as f:
    f.write(tflite_model)
    print("[MESSAGE] Model write to {}".format(args.out_tflite))
