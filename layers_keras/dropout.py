import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops


lib = tf.load_op_library('layers_cc/layers.so')
class DropoutNew(Layer):

    def __init__(self, rate, eid, seed=None, **kwargs):
        super(DropoutNew, self).__init__(**kwargs)
        self.rate = rate
        self.seed = seed
        if seed is None:
            self.seed = np.random.randint(10e6)
        self.eid = eid

    def build(self, input_shape):
        if not isinstance(self.rate, float) or (self.rate < 0 or self.rate >= 1):
            raise ValueError("rate must be a float in the range [0, 1), got %g" % rate)
        super(DropoutNew, self).build(input_shape)

    def call(self, x, training=None):
        if training is None:
            training = K.learning_phase()
        def drop():
            if self.rate == 0:
                random_seed.get_seed(self.seed)
                return x
            else:
                seed1, seed2 = random_seed.get_seed(self.seed)
            random_tensor = lib.uniform_distribution_new(x.shape, seed=seed1, seed2=seed2, dtype=x.dtype)
            return lib.dropout_new(input=x, random_tensor=random_tensor, rate=self.rate, eid_low=(self.eid&0xFFFFFFFF),eid_high=(self.eid>>32))
        output = control_flow_util.smart_cond(training, drop,
                                          lambda: array_ops.identity(x))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(DropoutNew, self).get_config()
        config.update({"rate": self.rate})
        config.update({"seed": self.seed})
        return config
        
@ops.RegisterGradient("DropoutNew")
def _dropout_new_grad_cc(op, grad):
    return [lib.dropout_new_grad(grad, op.inputs[1], rate=op.get_attr("rate"), eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high")),None]
    