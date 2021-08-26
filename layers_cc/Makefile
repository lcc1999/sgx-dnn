TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

objects = dense_kernel.cc dense_kernel_grad.cc\
relu_kernel.cc relu_kernel_grad.cc\
softmax_kernel.cc softmax_kernel_grad.cc\
dropout_kernel.cc dropout_kernel_grad.cc\
rand.cc

layers.so: $(objects)
	g++ -std=c++11 -shared $(objects) -o layers.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -DUSE_SGX -D_GLIBCXX_USE_CXX11_ABI=0 -O2

clean:
	rm -rf layers.so