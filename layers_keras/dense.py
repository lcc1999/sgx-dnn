import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops

import numpy as np
P = 2**23 + 2**21 + 7
MID = P // 2
assert(P + MID < 2**24)
q = float(round(np.sqrt(MID))) + 1
def mod_cast(low, high, q=q, p=P):
    tmp = tf.cast(low + q * high, tf.float64)
    return tf.cast(tf.math.floormod(tmp, p))

lib = tf.load_op_library('layers_cc/layers.so')
class DenseNew(Layer):

    def __init__(self, units, use_sgx=True, quantize=False, eid=None, prime=None,
            bits_w=None, bits_x=None, use_bias=True, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zero', **kwargs):
        self.units = units
        self.use_bias = use_bias
        self.quantize = quantize
        self.eid = eid
        self.prime = prime
        self.bits_w = bits_w
        self.bits_x = bits_x
        self.use_sgx = use_sgx
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.weight = None
        self.bias = None
        super(DenseNew, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight', 
                                      shape=(input_shape[1],self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias', 
                                      shape=(self.units,),
                                      initializer=self.bias_initializer,
                                      trainable=self.use_bias)
        super(DenseNew, self).build(input_shape)

    def call(self, x):
        if self.use_sgx:
            output = lib.dense_new(input=x, weights=self.weight, bias=self.bias, use_bias=self.use_bias, eid_low=(self.eid&0xFFFFFFFF),eid_high=(self.eid>>32))
        elif self.quantize:
            r = tf.cast(tf.random.uniform(shape=tf.shape(x),minval=-MID, maxval=MID+1,dtype=tf.int32),tf.float32)
            blind = gen_math_ops.MatMul(a=r, b=self.weight)
            x = x + r

            inputs_low = tf.math.floormod(x, q)
            inputs_high = tf.round((x - inputs_low) / q)
            outputs_low = gen_math_ops.MatMul(a=inputs_low, b=self.weight)
            outputs_high = gen_math_ops.MatMul(a=inputs_high, b=self.weight)
            if self.use_bias:
                outputs_low = nn_ops.bias_add(outputs_low, self.bias)
            output = tf.cast(outputs_low, tf.float64) + q * tf.cast(outputs_high, tf.float64)
            #output = tf.cast(tf.math.floormod(output, P), tf.float32)
            output = tf.cast(output, tf.float32)
            
            output = output - blind
            
            output=output/(2**self.bits_w)
        else:
            output = gen_math_ops.MatMul(a=x, b=self.weight)
            if self.use_bias:
                output = nn_ops.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.units)
        
    def get_config(self):
        config = super(DenseNew, self).get_config()
        config.update({"units": self.units})
        config.update({"use_bias": self.use_bias})
        config.update({"kernel_initializer":self.kernel_initializer})
        config.update({"bias_initializer":self.bias_initializer})
        config.update({"use_sgx":self.use_sgx,"eid":self.eid})
        config.update({"quantize":self.quantize,"prime":self.prime,"bits_w":self.bits_w,"bits_x":self.bits_x})
        return config
        

@ops.RegisterGradient("DenseNew")
def _dense_new_grad_cc(op, grad):
    return lib.dense_new_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], use_bias=op.get_attr("use_bias"), eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"))