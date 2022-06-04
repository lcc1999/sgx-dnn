import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers
import datetime
from tqdm import tqdm
from tensorflow.python.ops import clip_ops
from layers_keras import DenseNew,SoftmaxNew,ReluNew,DropoutNew
from utils import transform
from ctypes import *
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'


def prepare(x, y):
    x = tf.cast(x, tf.float32) / 255.
    y = tf.cast(y, tf.int32) 
    y = tf.one_hot(y,10)
    return x,y
(x_train, y_train),(x_test, y_test) = datasets.mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.map(prepare)
train_dataset = train_dataset.shuffle(60000).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataset = test_dataset.map(prepare)
test_dataset = test_dataset.shuffle(10000).batch(128)


def create_model():
  return tf.keras.models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    #DenseNew(units=256, eid=eid, kernel_initializer='glorot_uniform'),
    layers.Dense(512),
    #ReluNew(eid=eid),
    layers.ReLU(),
    #DropoutNew(rate=0.2, eid=eid),
    layers.Dropout(0.2),
    #DenseNew(units=10, eid=eid, kernel_initializer='glorot_uniform'),
    layers.Dense(10),
    #SoftmaxNew(eid=eid)
    layers.Softmax()
  ])
model = create_model()    

def loss_object(y_train, predictions):
    epsilon_ = 1e-7
    predictions = clip_ops.clip_by_value(predictions, epsilon_, 1. - epsilon_)
    return tf.reduce_mean(tf.reduce_sum(-y_train*tf.math.log(predictions),axis=-1))
#loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

def train_step(model, optimizer, x_train, y_train):
  with tf.GradientTape() as tape:
    predictions = model(x_train)
    loss = loss_object(y_train, predictions)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  train_loss(loss)
  train_accuracy(y_train, predictions)

def test_step(model, x_test, y_test):
  predictions = model(x_test)
  loss = loss_object(y_test, predictions)

  test_loss(loss)
  test_accuracy(y_test, predictions)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
test_log_dir = 'logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

EPOCHS = 1
start = datetime.datetime.now()
for epoch in range(EPOCHS):
  for (x_train, y_train) in train_dataset:
    train_step(model, optimizer, x_train, y_train)
  with train_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

  for (x_test, y_test) in test_dataset:
    test_step(model, x_test, y_test)
    
  with test_summary_writer.as_default():
    tf.summary.scalar('loss', test_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))
  train_loss.reset_states()
  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()
  print(datetime.datetime.now()-start)



sgx_lib_path = "layers_sgx.so"
sgx_lib = cdll.LoadLibrary(sgx_lib_path)
sgx_lib.initialize_enclave.restype = c_ulong
eid = sgx_lib.initialize_enclave()

### quantize=False,eid=eid
new_model=transform(model=model,input_shape=(1,28,28),dtype='float32',quantize=False,eid=eid)

### quantize=False,eid=None
#new_model=transform(model=model,input_shape=(1,28,28),dtype='float32',quantize=False,eid=None)
P = 2**23 + 2**21 + 7
### quantize=True,eid=eid
#new_model=transform(model=model,input_shape=(1,28,28),dtype='float32',quantize=True,eid=eid,prime=P,bits_w=8,bits_x=8)

### quantize=True,eid=None
#new_model=transform(model=model,input_shape=(1,28,28),dtype='float32',quantize=True,eid=None,prime=P,bits_w=8,bits_x=8)

print(new_model.get_config())

if True:
  for (x_test, y_test) in test_dataset:
    test_step(model, x_test, y_test)

  template = 'Test Loss: {}, Test Accuracy: {}'
  print (template.format(test_loss.result(), 
                         test_accuracy.result()*100))
  test_loss.reset_states()
  test_accuracy.reset_states()
  print(datetime.datetime.now()-start)
  
sgx_lib.destroy_enclave.argtypes = [c_ulong]
sgx_lib.destroy_enclave(eid)





'''
P = 2**23 + 2**21 + 7
bits_w=8
bits_x=8
def prepare(x, y):
    x = tf.math.round(tf.cast(x, tf.float32) / 255.*(2**bits_x))
    #x = tf.cast(x, tf.float32) / 255.
    y = tf.cast(y, tf.int32) 
    y = tf.one_hot(y,10)
    return x,y
(x_train, y_train),(x_test, y_test) = datasets.mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.map(prepare)
train_dataset = train_dataset.shuffle(60000).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataset = test_dataset.map(prepare)
test_dataset = test_dataset.shuffle(10000).batch(128)


new_model=transform(model=model,input_shape=(1,28,28),dtype='float32',quantize=True,eid=None,prime=P,bits_w=bits_w,bits_x=bits_x)

print(new_model.get_config())

if True:
  test_loss.reset_states()
  test_accuracy.reset_states()
  for (x_test, y_test) in test_dataset:
    test_step(new_model, x_test, y_test)
  template = 'Test Loss: {}, Test Accuracy: {}'
  print (template.format(test_loss.result(), 
                         test_accuracy.result()*100))
  test_loss.reset_states()
  test_accuracy.reset_states()
  print(datetime.datetime.now()-start)'''