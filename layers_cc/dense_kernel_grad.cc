#include <dlfcn.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("DenseNewGrad")
  .Attr("T: {float, double}")
  .Attr("use_bias: bool=true")
#ifdef USE_SGX
  .Attr("eid_low: int")
  .Attr("eid_high: int")
#endif
  .Input("grad: T")
  .Input("input: T")
  .Input("weights: T")
  .Input("bias: T")
  .Output("grad_input: T")
  .Output("grad_weights: T")
  .Output("grad_bias: T");

template <typename Device, typename T>
class DenseNewGradOp : public OpKernel {
public:
    explicit DenseNewGradOp(OpKernelConstruction* context) : OpKernel(context) {
#ifdef USE_SGX
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("layers_sgx.so", RTLD_NOW);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load layers_sgx.so!"));
#endif
        OP_REQUIRES_OK(context, context->GetAttr("use_bias", &use_bias_));
    }
  
    void Compute(OpKernelContext* context) override {
        DCHECK_EQ(context->num_inputs(),4);
        const Tensor& grad = context->input(0);
        const Tensor& input = context->input(1);
        const Tensor& weights = context->input(2);
        const Tensor& bias = context->input(3);
    
        const TensorShape& input_shape = input.shape();
        const TensorShape& weights_shape = weights.shape();
        const TensorShape& bias_shape = bias.shape();
        
        DCHECK_EQ(input_shape.dims(), 2);
        DCHECK_EQ(weights_shape.dims(), 2);
        DCHECK_EQ(bias_shape.dims(), 1);
        DCHECK_EQ(weights_shape.dim_size(0),input_shape.dim_size(1));
        DCHECK_EQ(bias_shape.dim_size(0),weights_shape.dim_size(1));
        
        DCHECK_EQ(input_shape.dim_size(0), grad.shape().dim_size(0));
        DCHECK_EQ(weights_shape.dim_size(1), grad.shape().dim_size(1));
    
        Tensor* grad_input = NULL;
        Tensor* grad_weights = NULL;
        Tensor* grad_bias = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
        OP_REQUIRES_OK(context, context->allocate_output(2, bias_shape, &grad_bias));
    
        /*auto grad_tensor = grad.matrix<T>();
        auto input_tensor = input.matrix<T>();
        auto weights_tensor = weights.matrix<T>();
        auto bias_tensor = bias.flat<T>();
        auto grad_input_tensor = grad_input->matrix<T>();
        auto grad_weights_tensor = grad_weights->matrix<T>();
        auto grad_bias_tensor = grad_bias->flat<T>();
        
        for(int i = 0; i < input_shape.dim_size(0); i++){
            for(int k = 0; k < input_shape.dim_size(1); k++){
                grad_input_tensor(i,k) = 0;
                for(int j = 0; j < weights_shape.dim_size(1); j++){
                    grad_input_tensor(i,k) += grad_tensor(i,j) * weights_tensor(k,j);
                }
            }
        }
        
        for(int k = 0; k < weights_shape.dim_size(0); k++){
            for(int j = 0; j < weights_shape.dim_size(1); j++){
                grad_weights_tensor(k,j) = 0;
                for(int i = 0; i < input_shape.dim_size(0); i++){
                    grad_weights_tensor(k,j) += grad_tensor(i,j) * input_tensor(i,k);
                }
            }
        }
        
        for(int j = 0; j < bias_shape.dim_size(0); j++){
            grad_bias_tensor(j) = 0;
            for(int i = 0; i < input_shape.dim_size(0); i++){
                grad_bias_tensor(j) += grad_tensor(i,j);
            }
        }*/
#ifdef USE_SGX
        long int dim_in[2] = {input_shape.dim_size(0), input_shape.dim_size(1)};
        long int dim_w[2] = {weights_shape.dim_size(0), weights_shape.dim_size(1)};
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float* input, float* weights, float* grad,
                                    float* grad_input, float* grad_weights, float* grad_bias, 
								  	long int dim_in[2], long int dim_w[2]);
        dlerror();

        function dense_backward = (function) dlsym(lib, "dense_backward");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of dense_backward failed: ", dlsym_error));
        dense_backward(eid_, (float*) input.flat<T>().data(), (float*) weights.flat<T>().data(), 
            (float*) grad.flat<T>().data(),(float*) grad_input->flat<T>().data(),
            (float*) grad_weights->flat<T>().data(),(float*) grad_bias->flat<T>().data(), dim_in, dim_w);
#else
        auto g=grad.tensor<T, 2>();
        auto x=input.tensor<T, 2>();
        auto w=weights.tensor<T, 2>();
        Eigen::array<Eigen::IndexPair<int>, 1> gx_dim = { Eigen::IndexPair<int>(1, 1) };
        Eigen::array<Eigen::IndexPair<int>, 1> gw_dim = { Eigen::IndexPair<int>(0, 0) };
        Eigen::array<int, 1> gb_dims({0});
        grad_input->tensor<T, 2>() = g.contract(w,gx_dim);
        grad_weights->tensor<T, 2>() = x.contract(g,gw_dim);
        grad_bias->tensor<T, 1>() = g.sum(gb_dims);
#endif
    }
  
private:
    bool use_bias_;
    void* lib;
#ifdef USE_SGX
    int64 eid_low_;
    int64 eid_high_;
#endif
};

#ifndef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DenseNewGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        DenseNewGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("DenseNewGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        DenseNewGradOp<CPUDevice, double>);
#endif