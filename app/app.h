#include <iostream>
#include "sgx_urts.h"

#define ENCLAVE_FILENAME "libenclave.signed.so"
extern "C"{
    unsigned long int initialize_enclave();
    void destroy_enclave(unsigned long int eid);
    void dense(unsigned long int eid, float* input, float* output, float* weights, float* bias, long int dim_in[2], long int dim_w[2]);
    void dense_backward(unsigned long int eid, float* input, float* weights, float* grad, float* grad_input, float* grad_weights, float* grad_bias, long int dim_in[2], long int dim_w[2]);
    void relu(unsigned long int eid, float* input, float* output, long int dim);
    void relu_backward(unsigned long int eid, float* input, float* grad, float* grad_input, long int dim);
    void softmax(unsigned long int eid, float* input, float* output, long int dim[2]);
    void softmax_backward(unsigned long int eid, float* softmax, float* grad, float* grad_input, long int dim[2]);
    void dropout(unsigned long int eid, float* input, float* random, float* output, long int dim, float rate);
    void dropout_backward(unsigned long int eid, float* random, float* grad, float* grad_input, long int dim, float rate);
}
