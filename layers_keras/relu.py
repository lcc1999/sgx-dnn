import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn

lib = tf.load_op_library('layers_cc/layers.so')
class ReluNew(Layer):

    def __init__(self, use_sgx=True, quantize=False, eid=None, 
            prime=None, bits_w=None, bits_x=None, **kwargs):
        self.quantize = quantize
        self.eid = eid
        self.prime = prime
        self.bits_w = bits_w
        self.bits_x = bits_x
        self.use_sgx = use_sgx
        super(ReluNew, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReluNew, self).build(input_shape)

    def call(self, x):
        if self.use_sgx:
            output = lib.relu_new(input=x, eid_low=(self.eid&0xFFFFFFFF),eid_high=(self.eid>>32))
        elif self.quantize:
            output = nn.relu(x)
        else:
            output = nn.relu(x)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(ReluNew, self).get_config()
        config.update({"use_sgx":self.use_sgx,"eid":self.eid})
        config.update({"quantize":self.quantize,"prime":self.prime,"bits_w":self.bits_w,"bits_x":self.bits_x})
        return config
        

@ops.RegisterGradient("ReluNew")
def _relu_new_grad_cc(op, grad):
    return lib.relu_new_grad(grad, op.inputs[0], eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"))