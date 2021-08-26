#include <dlfcn.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("SoftmaxNewGrad")
  .Attr("T: {float, double}")
#ifdef USE_SGX
  .Attr("eid_low: int")
  .Attr("eid_high: int")
#endif
  .Input("grad: T")
  .Input("softmax: T")
  .Output("grad_input: T");

template <typename Device, typename T>
class SoftmaxNewGradOp : public OpKernel {
public:
    explicit SoftmaxNewGradOp(OpKernelConstruction* context) : OpKernel(context) {
#ifdef USE_SGX
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("layers_sgx.so", RTLD_NOW);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load layers_sgx.so!"));
#endif
    }
  
    void Compute(OpKernelContext* context) override {
        DCHECK_EQ(context->num_inputs(),2);
        const Tensor& grad = context->input(0);
        const Tensor& softmax = context->input(1);
        const TensorShape& grad_shape = grad.shape();
        const TensorShape& input_shape = softmax.shape();
        DCHECK_EQ(input_shape.dims(), 2);
        DCHECK_EQ(input_shape, grad_shape);
        
        Tensor* grad_input = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
        
#ifdef USE_SGX
        long int dim[2] = {input_shape.dim_size(0), input_shape.dim_size(1)};
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float* softmax, float* grad, float* grad_input, long int dim[2]);
        dlerror();

        function softmax_backward = (function) dlsym(lib, "softmax_backward");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of softmax_backward failed: ", dlsym_error));
        softmax_backward(eid_, (float*) softmax.flat<T>().data(), (float*) grad.flat<T>().data(), (float*) grad_input->flat<T>().data(), dim);
#else
        auto y = softmax.tensor<T, 2>();
        auto g = grad.tensor<T, 2>();
        const int batch_size = y.dimension(0);
        const int num_classes = y.dimension(1);
        
        Eigen::DSizes<int, 1> along_class(1);
        Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
        Eigen::DSizes<int, 2> one_by_class(1, num_classes);

        grad_input->tensor<T, 2>() = y * (g - (y * g).sum(along_class)
                                                    .eval().reshape(batch_by_one)
                                                    .broadcast(one_by_class));
#endif
    }
private:
    void* lib;
#ifdef USE_SGX
    int64 eid_low_;
    int64 eid_high_;
#endif
};

#ifndef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("SoftmaxNewGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        SoftmaxNewGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SoftmaxNewGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        SoftmaxNewGradOp<CPUDevice, double>);
#endif