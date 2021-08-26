#include <dlfcn.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("ReluNewGrad")
  .Attr("T: {float, double}")
#ifdef USE_SGX
  .Attr("eid_low: int")
  .Attr("eid_high: int")
#endif
  .Input("grad: T")
  .Input("input: T")
  .Output("grad_input: T");

template <typename Device, typename T>
class ReluNewGradOp : public OpKernel {
public:
    explicit ReluNewGradOp(OpKernelConstruction* context) : OpKernel(context) {
#ifdef USE_SGX
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("layers_sgx.so", RTLD_NOW);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load layers_sgx.so!"));
#endif
    }
  
    void Compute(OpKernelContext* context) override {
        const Tensor& grad = context->input(0);
        const Tensor& input = context->input(1);
        const TensorShape& shape = input.shape();
        const TensorShape& g_shape = grad.shape();
        DCHECK_EQ(shape, g_shape);
        Tensor* grad_input = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &grad_input));
        
        auto x = input.template flat<T>();
        auto g = grad.template flat<T>();
#ifdef USE_SGX
        long int dim = x.size();
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float* input, float* grad, float* grad_input, long int dim);
        dlerror();

        function relu_backward = (function) dlsym(lib, "relu_backward");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of relu_backward failed: ", dlsym_error));
        relu_backward(eid_, (float*) x.data(), (float*) g.data(), (float*) grad_input->flat<T>().data(), dim);
#else
        grad_input->template flat<T>() = g * (x > static_cast<T>(0)).template cast<T>();
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
REGISTER_KERNEL_BUILDER(Name("ReluNewGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ReluNewGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("ReluNewGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        ReluNewGradOp<CPUDevice, double>);
#endif