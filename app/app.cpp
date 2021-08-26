#include <string.h>
#include <assert.h>
#include <fstream>
#include <thread>
#include <iostream>

#include "sgx_urts.h"
#include "Enclave_u.h"

#include "ErrorSupport.h"
#include "app.h"

void ocall_print_string(const char *str)
{
  printf("%s", str);
}

unsigned long int initialize_enclave()
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    unsigned long int eid;
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &eid, NULL);
    if (ret != SGX_SUCCESS) {
        ret_error_support(ret);
        throw ret;
    }
    printf("initialize %lu finish.\n",eid);
    return eid;
}

void destroy_enclave(unsigned long int eid)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    ret = sgx_destroy_enclave(eid);
    if (ret != SGX_SUCCESS) {
        ret_error_support(ret);
        throw ret;
    }
    printf("destroy %lu finish.\n",eid);
}

void dense(unsigned long int eid, float* input, float* output, float* weights, float* bias, 
        long int dim_in[2], long int dim_w[2])
{
	sgx_status_t ret = ecall_dense(eid, input, output, weights, bias, dim_in, dim_w);
	if (ret != SGX_SUCCESS) {
		ret_error_support(ret);
		throw ret;
	}
}

void dense_backward(unsigned long int eid, float* input, float* weights, float* grad, float* grad_input, 
            float* grad_weights, float* grad_bias, long int dim_in[2], long int dim_w[2])
{
	sgx_status_t ret = ecall_dense_backward(eid, input, weights, grad, grad_input, grad_weights, 
                    grad_bias, dim_in, dim_w);
	if (ret != SGX_SUCCESS) {
		ret_error_support(ret);
		throw ret;
	}
}

void relu(unsigned long int eid, float* input, float* output, long int dim){
    sgx_status_t ret = ecall_relu(eid, input, output, dim);
	if (ret != SGX_SUCCESS) {
		ret_error_support(ret);
		throw ret;
	}
}
void relu_backward(unsigned long int eid, float* input, float* grad, float* grad_input, long int dim){
    sgx_status_t ret = ecall_relu_backward(eid, input, grad, grad_input, dim);
	if (ret != SGX_SUCCESS) {
		ret_error_support(ret);
		throw ret;
	}
}

void softmax(unsigned long int eid, float* input, float* output, long int dim[2]){
    sgx_status_t ret = ecall_softmax(eid, input, output, dim);
	if (ret != SGX_SUCCESS) {
		ret_error_support(ret);
		throw ret;
	}
}

void softmax_backward(unsigned long int eid, float* softmax, float* grad, float* grad_input, long int dim[2]){
    sgx_status_t ret = ecall_softmax_backward(eid, softmax, grad, grad_input, dim);
	if (ret != SGX_SUCCESS) {
		ret_error_support(ret);
		throw ret;
	}
}

void dropout(unsigned long int eid, float* input, float* random, float* output, long int dim, float rate){
    sgx_status_t ret = ecall_dropout(eid, input, random, output, dim, rate);
	if (ret != SGX_SUCCESS) {
		ret_error_support(ret);
		throw ret;
	}
}

void dropout_backward(unsigned long int eid, float* random, float* grad, float* grad_input, long int dim, float rate){
    sgx_status_t ret = ecall_dropout_backward(eid, random, grad, grad_input, dim, rate);
	if (ret != SGX_SUCCESS) {
		ret_error_support(ret);
		throw ret;
	}
}
