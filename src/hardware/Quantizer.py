import math
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_scope
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_apply
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_layer
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantize_layer import QuantizeLayer
from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_impl import Pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay
import inspect
import tensorflow_model_optimization as tfmot
from tensorflow import keras


def get_quant_exp(variable, num_bits, return_abs=False):
    abs_variable = tf.abs(variable)
    max_variable = tf.reduce_max(abs_variable)

    # we need a conditional statement because of the array is entirely 0 we would get a -inf log
    log_variable = tf.cond(
        tf.equal(max_variable, 0.0),
        lambda: tf.constant(0.0, variable.dtype),
        lambda: tf.math.log(max_variable) / tf.constant(math.log(2.0), variable.dtype)  # tf doesnt have log2, so we compute log in base e and then divide to cast it to log2
    )

    log_rounded = tf.cast(tf.math.ceil(log_variable), tf.int8)
    unsigned_width = num_bits - 1  # for the sign bit
    exp_diff = unsigned_width - log_rounded
    no_grad_exp = tf.stop_gradient(exp_diff)
    if return_abs is False:
        return no_grad_exp
    else:
        return no_grad_exp, abs_variable


def quantize_variable(variable, exp, width, clip, round_or_floor, name):
    # Use tf.custom_gradient to add a gradient to round and floor operations
    # We need a sub function because @tf.custom_gradient doesnt support kwargs
    @tf.custom_gradient
    def _quantize_variable(l_var):
        def grad(dy):  # identity gradient
            return dy

        # Round doesnt have a gradient, we force it to identity
        quant_shift = 2.0 ** tf.cast(exp, tf.float32)  # cast for ** compatiblity
        quantizing = l_var * quant_shift

        if round_or_floor == "round":
            quantizing = tf.round(quantizing, name=name + "_round")
        elif round_or_floor == "floor":
            quantizing = tf.floor(quantizing, name=name + "_floor")
        else:
            raise ValueError("Illegal round mode {}".format(round_or_floor))

        if clip is True:
            max_value = 2.0 ** (width - 1)
            quantizing = tf.clip_by_value(quantizing, -max_value, max_value - 1, name=name + "_clip")

        quantizing = quantizing / quant_shift
        return quantizing, grad

    quantized_variable = _quantize_variable(variable)
    return quantized_variable


# base class not to be used
class BFPQuantizer(Quantizer):

    def __init__(self, num_bits):
        self.num_bits = num_bits

    def get_config(self):
        config = dict()
        config["num_bits"] = self.num_bits
        return config

    def build(self, tensor_shape, name, layer):
        variable_dict = dict()

        variable_dict["exp"] = layer.add_weight(
            name=name + "_exp",
            dtype=tf.int8,
            initializer=keras.initializers.Constant(value=self.num_bits),  # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
            trainable=False
        )

        return variable_dict


class BFPWeightQuantizer(BFPQuantizer):
    # Pruning type furnished either as class type and not an object
    # Pruning config furnished as dictionary
    def __init__(self, num_bits, enable_pruning=False, pruning_type=None, pruning_config=None):
        super().__init__(num_bits)

        self.enable_pruning = enable_pruning
        self.pruning_type = pruning_type

        if pruning_config is not None:
            self.pruning_config = pruning_config
        else:
            self.pruning_config = {
                "initial_sparsity": 0.0,
                "final_sparsity": 0.0,
                "begin_step": 10,
                "end_step": 1500 * 20,
                "frequency": 100
            }

    def get_config(self):
        config = super().get_config()
        config["pruning_config"] = self.pruning_config
        config["pruning_type"] = self.pruning_type
        config["enable_pruning"] = self.enable_pruning
        return config

    def build(self, tensor_shape, name, layer):
        variable_dict = super().build(tensor_shape, name, layer)

        variable_dict["stored_tensor"] = layer.add_weight(
            name=name + "_stored_tensor",
            shape=tensor_shape,
            initializer=keras.initializers.glorot_normal(),  # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
            trainable=False
        )
        if self.enable_pruning is True:
            # Pruning variables
            variable_dict["mask"] = layer.add_weight(
                'mask',
                shape=tensor_shape,
                initializer=tf.keras.initializers.get('zeros'),
                dtype=variable_dict["stored_tensor"].dtype,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN)

            variable_dict["threshold"] = layer.add_weight(
                'threshold',
                shape=[],
                initializer=tf.keras.initializers.get('ones'),
                dtype=variable_dict["stored_tensor"].dtype,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN)

            variable_dict["pruning_step"] = layer.add_weight(
                'pruning_step',
                shape=[],
                initializer=tf.keras.initializers.Constant(-1),
                dtype=tf.int64,
                trainable=False)

            self.num_weights_int = tf.size(variable_dict["stored_tensor"])
            self.num_weights_fp = tf.dtypes.cast(self.num_weights_int, tf.float32)

            if self.pruning_type is None:
                self.pruning_schedule = PolynomialDecay(**self.pruning_config)
            else:
                self.pruning_schedule = self.pruning_type(**self.pruning_config)

        return variable_dict

    def __call__(self, inputs, training, weights, **kwargs):

        # Weights mode and we aren't in training
        # don't need to be re quantized outside training
        if training is False:
            return weights["stored_tensor"]
        else:
            assign_ops = []
            if self.enable_pruning is True:
                # Get quantization exp and absoluted array
                quant_exp, abs_weights = get_quant_exp(inputs, self.num_bits, return_abs=True)

                def update_mask(sparsity):
                    # Compute position of threshold in the array. Must be capped to avoid error when sparsity is 0
                    pruning_threshold_index_uncapped = tf.dtypes.cast(tf.math.round(self.num_weights_fp * (1 - sparsity)), tf.int32)
                    pruning_threshold_index = tf.math.minimum(pruning_threshold_index_uncapped, self.num_weights_int - 1)

                    # Sort the entire array (flattened)
                    sorted_weights, _ = tf.math.top_k(tf.reshape(abs_weights, [-1]), k=self.num_weights_int)

                    # Selected the threshold value
                    current_threshold = sorted_weights[pruning_threshold_index]

                    # compare to get new mask
                    mask_bool = tf.math.greater_equal(abs_weights, current_threshold)
                    mask = tf.dtypes.cast(mask_bool, inputs.dtype)

                    assign_ops.append(weights["mask"].assign(mask))
                    assign_ops.append(weights["threshold"].assign(current_threshold))

                    return mask

                # Get new sparsity level
                update_pruning, sparsity = self.pruning_schedule(weights["pruning_step"])
                sel_mask = tf.cond(update_pruning, lambda: update_mask(sparsity), lambda: weights["mask"])
                no_grad_mask = tf.stop_gradient(sel_mask)
                masked_weight = tf.math.multiply(inputs, no_grad_mask)
                incremented_pruning_step = weights["pruning_step"] + 1
                assign_ops.append(weights["pruning_step"].assign(incremented_pruning_step))  # updated even if not pruning since we need it for keeping track of when to prune
            else:
                # Get quantization exp and absoluted array
                quant_exp = get_quant_exp(inputs, self.num_bits, return_abs=False)
                masked_weight = inputs

            # Quantization
            quantized_inputs = quantize_variable(variable=masked_weight, exp=quant_exp, width=self.num_bits, clip=False, round_or_floor="round", name="weight")

            assign_ops.append(weights["stored_tensor"].assign(quantized_inputs))
            assign_ops.append(weights["exp"].assign(quant_exp))

            with tf.control_dependencies(assign_ops):
                quantized_inputs = tf.identity(quantized_inputs)

            return quantized_inputs


class BFPActivQuantizer(BFPQuantizer):
    def __init__(self, num_bits, num_batch):
        super().__init__(num_bits)
        self.num_batch = num_batch
        self.exp = num_bits

    def get_config(self):
        config = super().get_config()
        config["num_batch"] = self.num_batch
        return config

    def build(self, tensor_shape, name, layer):
        variable_dict = super().build(tensor_shape, name, layer)

        variable_dict["exp_memory"] = layer.add_weight(
            name=name + "_exp_memory",
            dtype=tf.int8,
            shape=(self.num_batch,),
            initializer=keras.initializers.Constant(value=0),  # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
            trainable=False
        )
        variable_dict["exp_memory_ptr"] = layer.add_weight(
            name=name + "_exp_memory_ptr",
            dtype=tf.int32,
            initializer=keras.initializers.Constant(value=0),  # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
            trainable=False
        )

        self.exp = variable_dict["exp"]  # for bias quantization

        return variable_dict

    def __call__(self, inputs, training, weights, **kwargs):
        if training is True:
            new_exp = get_quant_exp(inputs, self.num_bits)
            new_memory_ptr = tf.math.floormod(weights["exp_memory_ptr"] + 1, self.num_batch)

            exp_memory_assign_op = weights["exp_memory"][weights["exp_memory_ptr"]].assign(new_exp)
            exp_memory_ptr_assign_op = weights["exp_memory_ptr"].assign(new_memory_ptr)

            quant_exp = tf.reduce_max(weights["exp_memory"])
            exp_assign_op = weights["exp"].assign(quant_exp)

            self.exp = quant_exp #used by the bias quantizer

            clip = False
            with tf.control_dependencies([exp_memory_assign_op, exp_memory_ptr_assign_op, exp_memory_ptr_assign_op, exp_assign_op]):
                quantized_inputs = quantize_variable(variable=inputs, exp=quant_exp, width=self.num_bits, clip=clip, round_or_floor="floor", name="activ")
        else:
            quant_exp = weights["exp"]
            clip = True
            quantized_inputs = quantize_variable(variable=inputs, exp=quant_exp, width=self.num_bits, clip=clip, round_or_floor="floor", name="activ")

        return quantized_inputs


class BFPBiasQuantizer(BFPQuantizer):
    def __init__(self, num_bits, activ_quantizer):
        super().__init__(num_bits)
        self.activ_quantizer = activ_quantizer

    def get_config(self):
        config = super().get_config()
        config["activ_quantizer"] = self.activ_quantizer
        return config

    def build(self, tensor_shape, name, layer):
        variable_dict = super().build(tensor_shape, name, layer)

        variable_dict["stored_tensor"] = layer.add_weight(
            name=name + "_stored_tensor",
            shape=tensor_shape,
            initializer=keras.initializers.zeros(),  # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
            trainable=False
        )

        return variable_dict

    def __call__(self, inputs, training, weights, **kwargs):

        # Weights mode and we aren't in training
        # don't need to be re quantized outside training
        if training is False:
            return weights["stored_tensor"]
        else:
            exp_assign_op = weights["exp"].assign(self.activ_quantizer.exp)
            quantized_inputs = quantize_variable(variable=inputs, exp=exp_assign_op, width=self.num_bits, clip=True, round_or_floor="round", name="bias")
            stored_tensor_assign_op = weights["stored_tensor"].assign(quantized_inputs)

            with tf.control_dependencies([stored_tensor_assign_op]):
                quantized_inputs = tf.identity(quantized_inputs)

            return quantized_inputs


class BFPQuantizeConfig(QuantizeConfig):
    def __init__(self, output_quantizer=None, weight_quantizer=None, bias_quantizer=None):
        if output_quantizer is None:
            self.output_quantizer = BFPActivQuantizer(num_bits=16, num_batch=int(2 ** 12))
        else:
            self.output_quantizer = output_quantizer

        if weight_quantizer is None:
            self.weight_quantizer = BFPWeightQuantizer(num_bits=8)
        else:
            self.weight_quantizer = weight_quantizer

        if bias_quantizer is None:
            self.bias_quantizer = BFPBiasQuantizer(num_bits=8, activ_quantizer=self.output_quantizer)
        else:
            self.bias_quantizer = bias_quantizer

    # Configure how to quantize weights and biases
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, self.weight_quantizer),
                (layer.bias, self.bias_quantizer)
                ]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in get_weights_and_quantizers in the same order
        layer.kernel = quantize_weights[0]
        layer.bias = quantize_weights[1]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`, in the same order.
        pass

    # Configure how to quantize outputs.
    def get_output_quantizers(self, layer):
        return [self.output_quantizer]

    def get_config(self):
        config = dict()
        config["output_quantizer"] = self.output_quantizer
        config["weight_quantizer"] = self.weight_quantizer
        config["bias_quantizer"] = self.bias_quantizer
        return config


def apply_quantization(model):
    # Helper function uses `quantize_annotate_layer` to annotate that only the
    # Dense layers should be quantized.
    def add_quantize_annotation(layer):
        quantization_map = {
            tf.keras.layers.Dense: BFPQuantizeConfig(),
            tf.keras.layers.Conv2D: BFPQuantizeConfig()
        }

        for layer_type, quantize_config in quantization_map.items():
            if isinstance(layer, layer_type):
                print("**Quantization annotation added to layer {} of type {} with {}".format(layer.name, layer_type, quantize_config))
                return quantize_annotate_layer(to_annotate=layer, quantize_config=quantize_config)
        print("**Quantization annotation not added to layer {} of type {}".format(layer.name, type(layer)))
        return layer

    # Use `tf.keras.models.clone_model` to apply `add_quantize_annotation`
    # to the layers of the model.
    print("Annotating model {}".format(model.name))
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=add_quantize_annotation,
    )

    with quantize_scope({
        'BFPQuantizeConfig': BFPQuantizeConfig,
        "BFPActivQuantizer": BFPActivQuantizer,
        "BFPWeightQuantizer": BFPWeightQuantizer,
        "BFPBiasQuantizer": BFPBiasQuantizer,
        "PolynomialDecay": PolynomialDecay
    }):
        # Use `quantize_apply` to actually make the model quantization aware.
        quant_aware_model = quantize_apply(annotated_model)
        return quant_aware_model
