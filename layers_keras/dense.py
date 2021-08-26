import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops

lib = tf.load_op_library('layers_cc/layers.so')
class DenseNew(Layer):

    def __init__(self, units, eid, use_bias=True, kernel_initializer='glorot_uniform', **kwargs):
        self.units = units
        self.use_bias = use_bias
        self.eid = eid
        self.kernel_initializer = kernel_initializer
        super(DenseNew, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight', 
                                      shape=(input_shape[1],self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias', 
                                      shape=(self.units,),
                                      initializer='zero',
                                      trainable=self.use_bias)
        super(DenseNew, self).build(input_shape)

    def call(self, x):
        output = lib.dense_new(input=x, weights=self.weight, bias=self.bias, use_bias=self.use_bias, eid_low=(self.eid&0xFFFFFFFF),eid_high=(self.eid>>32))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.units)
        
    def get_config(self):
        config = super(DenseNew, self).get_config()
        config.update({"units": self.units})
        config.update({"use_bias": self.use_bias})
        return config
        

@ops.RegisterGradient("DenseNew")
def _dense_new_grad_cc(op, grad):
    return lib.dense_new_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], use_bias=op.get_attr("use_bias"), eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"))