"""Running Edge TPU model.

Borrowed some functions from:
google-coral/tflite/blob/master/python/examples/classification/classify.py

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import time

import numpy as np
import tflite_runtime.interpreter as tflite


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, data):
    """Copies data to input tensor."""
    # left
    interpreter.set_tensor(input_details[0]['index'], data[0])
    # right
    interpreter.set_tensor(input_details[1]['index'], data[1])


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--count", type=int, default=10)

args = parser.parse_args()

# get a interpreter
interpreter = tflite.Interpreter(
    args.model_path,
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)


input_shape = [1, 64, 64, 3]

sample_input_1 = np.random.normal(size=input_shape, dtype=np.float32)
sample_input_2 = np.random.normal(size=input_shape, dtype=np.float32)

set_input(interpreter, (sample_input_1, sample_input_2))


# measure time

for _ in range(args.count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start

    print('%.1fms' % (inference_time * 1000))
