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
    try:
        interpreter.set_tensor(input_details[1]['index'], data[1])
    except Exception:
        pass


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--count", type=int, default=100)

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

input_shape = input_details[0]['shape']
sample_input_1 = np.array(
    np.random.random_sample(input_shape), dtype=np.float32)
sample_input_2 = np.array(
    np.random.random_sample(input_shape), dtype=np.float32)

set_input(interpreter, (sample_input_1, sample_input_2))


# measure time
time_collector = []
for i in range(args.count+100):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start

    print('%.1fms' % (inference_time * 1000), end='\r')

    if i >= 100:
        time_collector.append(inference_time*1000)

print("-"*50)
print("Inference Time Mean: {}ms".format(np.mean(time_collector)))
print("Inference Time Std: {}ms".format(np.std(time_collector)))
print("-"*50)
