import numpy as np
import tensorflow as tf
from layers_keras import DenseNew,SoftmaxNew,ReluNew,DropoutNew

def transform(model,input_shape,dtype,quantize,eid,prime=None,bits_w=None,bits_x=None):
    old_layer = (tf.keras.layers.Dense, tf.keras.layers.ReLU, 
                tf.keras.layers.Dropout,tf.keras.layers.Softmax)
    weight_layer = (tf.keras.layers.Dense,)
    new_model = tf.keras.models.Sequential()
    x = tf.keras.layers.Input(shape=input_shape, dtype=dtype)
    for i,layer in enumerate(model.layers):
        if isinstance(layer, old_layer):
            config=layer.get_config()
            config["quantize"]=quantize
            config["eid"]=eid
            config["use_sgx"]=False if eid==None else True
            config["prime"]=prime
            config["bits_w"]=bits_w
            config["bits_x"]=bits_x
            common_keys=['name', 'trainable', 'dtype', 'use_sgx', 'eid', 'quantize', 'prime', 'bits_w', 'bits_x']
            if isinstance(layer,tf.keras.layers.Dense):
                keys=common_keys+['units', 'use_bias', 'kernel_initializer', 'bias_initializer']
                config=dict((key,value) for key,value in config.items() if key in keys)
                new_layer = DenseNew.from_config(config)
            elif isinstance(layer,tf.keras.layers.ReLU):
                keys=common_keys
                config=dict((key,value) for key,value in config.items() if key in keys)
                new_layer = ReluNew.from_config(config)
            elif isinstance(layer,tf.keras.layers.Dropout):
                keys=common_keys+['rate', 'seed']
                config=dict((key,value) for key,value in config.items() if key in keys)
                new_layer = DropoutNew.from_config(config)
            elif isinstance(layer,tf.keras.layers.Softmax):
                keys=common_keys
                config=dict((key,value) for key,value in config.items() if key in keys)
                new_layer = SoftmaxNew.from_config(config)
            x=new_layer(x)
            if isinstance(layer,weight_layer):
                kernel,bias=layer.get_weights()
                if quantize:
                    kernel,bias=np.round(kernel*2**bits_w),np.round(bias*2**bits_w*bits_x)
                new_layer.set_weights((kernel,bias))
            new_model.add(new_layer)
        else:
            x=layer(x)
            new_model.add(layer)
    print(new_model.summary())
    return new_model