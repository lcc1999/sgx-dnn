import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops

lib = tf.load_op_library('layers_cc/layers.so')
class SoftmaxNew(Layer):

    def __init__(self, eid, **kwargs):
        self.eid = eid
        super(SoftmaxNew, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SoftmaxNew, self).build(input_shape)

    def call(self, x):
        output = lib.softmax_new(input=x, eid_low=(self.eid&0xFFFFFFFF),eid_high=(self.eid>>32))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(SoftmaxNew, self).get_config()
        return config
        

@ops.RegisterGradient("SoftmaxNew")
def _softmax_new_grad_cc(op, grad):
    return lib.softmax_new_grad(grad, op.outputs[0], eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"))