"""Running Edge TPU model.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse

import tflite_runtime.interpreter as tflite

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)

args = parser.parse_args()

# get a interpreter
interpreter = tflite.Interpreter(
    args.model_path,
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()

_, height, width, dim = interpreter.get_input_details()[0]['shape']
print(height, width, dim)
