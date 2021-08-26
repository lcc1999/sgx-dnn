#include <dlfcn.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("DropoutNewGrad")
  .Attr("T: {float, double}")
  .Attr("rate: float")
#ifdef USE_SGX
  .Attr("eid_low: int")
  .Attr("eid_high: int")
#endif
  .Input("grad: T")
  .Input("random_tensor: T")
  .Output("grad_input: T");

template <typename Device, typename T>
class DropoutNewGradOp : public OpKernel {
public:
    explicit DropoutNewGradOp(OpKernelConstruction* context) : OpKernel(context) {
#ifdef USE_SGX
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("layers_sgx.so", RTLD_NOW);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load layers_sgx.so!"));
#endif
        OP_REQUIRES_OK(context, context->GetAttr("rate", &rate_));
        keep_prob = 1 - rate_;
        scale = 1 / keep_prob;
    }
  
    void Compute(OpKernelContext* context) override {
        DCHECK_EQ(context->num_inputs(),2);
        const Tensor& grad = context->input(0);
        const Tensor& random_tensor = context->input(1);
        const TensorShape& shape = grad.shape();
        
        Tensor* grad_input = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &grad_input));
        
        auto g = grad.template flat<T>();
        auto random = random_tensor.template flat<T>();
#ifdef USE_SGX
        long int dim = random.size();
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float* random, float* grad, float* grad_input, long int dim, float rate);
        dlerror();

        function dropout_backward = (function) dlsym(lib, "dropout_backward");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of dropout_backward failed: ", dlsym_error));
        dropout_backward(eid_, (float*) random.data(), (float*) g.data(), (float*) grad_input->flat<T>().data(), dim, rate_);
#else        
        auto keep_mask = (random >= static_cast<T>(rate_)).template cast<T>();
        grad_input->template flat<T>() = g*keep_mask*static_cast<T>(scale);
#endif
    }
  
private:
    float rate_;
    float keep_prob;
    float scale;
    void* lib;
#ifdef USE_SGX
    int64 eid_low_;
    int64 eid_high_;
#endif
};

#ifndef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DropoutNewGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        DropoutNewGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("DropoutNewGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        DropoutNewGradOp<CPUDevice, double>);
#endif