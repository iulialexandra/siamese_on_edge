"""Import trained model, split, and save them separately.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os
import argparse

import tensorflow as tf

from networks.edge_tpu_nets import HorizontalNetworkOnEdge


def convert_to_tflite(model, input_array, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        num_calibration_steps = 10
        for _ in range(num_calibration_steps):
            yield input_array

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8

    print("[MESSAGE] Converting file")
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(out_path, "wb") as f:
        f.write(tflite_model)
        print("[MESSAGE] Model write to {}".format(out_path))


# force to use CPU
tf.config.set_visible_devices([], 'GPU')

parser = argparse.ArgumentParser()

parser.add_argument("--keras_model", type=str, default="")
parser.add_argument("--out_dir", type=str)
parser.add_argument('--input_shape', nargs='+', type=int)

args = parser.parse_args()

input_shape = list(args.input_shape)

if args.keras_model != "":
    net = HorizontalNetworkOnEdge(input_shape)

    model = net.build_edge_net(None)

    # load weights
    model.load_weights(args.keras_model)
    print("[MESSAGE] Model is loaded")
    model.summary()


image = tf.random.normal([1]+input_shape)
feature = tf.random.normal([1, 8, 8, 96])
# split them
# branch
convert_to_tflite(net.embedding_model, [image],
                  os.path.join(args.out_dir, "branch.tflite"))

# trunk
convert_to_tflite(net.trunk_model, [feature],
                  os.path.join(args.out_dir, "trunk.tflite"))
