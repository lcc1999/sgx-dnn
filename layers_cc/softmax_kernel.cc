#include <dlfcn.h>
#include "tensorflow/core/framework/op.h" 
#include "tensorflow/core/framework/op_kernel.h" 
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("SoftmaxNew")
  .Attr("T: {float, double}")
#ifdef USE_SGX
  .Attr("eid_low: int")
  .Attr("eid_high: int")
#endif
  .Input("input: T")
  .Output("softmax: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  })
  .Doc(R"doc(  
    Softmax layer.
  )doc");


template <typename Device, typename T>
class SoftmaxNewOp : public OpKernel {
public:
    explicit SoftmaxNewOp(OpKernelConstruction* context) : OpKernel(context) {
#ifdef USE_SGX
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("layers_sgx.so", RTLD_NOW);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load layers_sgx.so!"));
#endif
    }

    void Compute(OpKernelContext* context) override {
        DCHECK_EQ(context->num_inputs(),1);
        const Tensor& input = context->input(0);
        const TensorShape& input_shape = input.shape();
        DCHECK_EQ(input_shape.dims(), 2);
        
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        
#ifdef USE_SGX
        long int dim[2] = {input_shape.dim_size(0), input_shape.dim_size(1)};
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float* input, float* output, long int dim[2]);
        dlerror();
        
        function softmax = (function) dlsym(lib, "softmax");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of softmax failed: ", dlsym_error));
        softmax(eid_, (float*) input.flat<T>().data(), (float*) output->flat<T>().data(), dim);
#else
        auto logits = input.tensor<T, 2>();
        const int batch_size = logits.dimension(0);
        const int num_classes = logits.dimension(1);
        
        Eigen::DSizes<int, 1> along_class(1);
        Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
        Eigen::DSizes<int, 2> one_by_class(1, num_classes);
        
        auto shifted_logits = (logits - logits.maximum(along_class)
                                        .eval()
                                        .reshape(batch_by_one)
                                        .broadcast(one_by_class)).exp();
        output->tensor<T, 2>() = (shifted_logits * shifted_logits.sum(along_class)
                                         .inverse()
                                         .eval()
                                         .reshape(batch_by_one)
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
REGISTER_KERNEL_BUILDER(Name("SoftmaxNew").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        SoftmaxNewOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SoftmaxNew").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        SoftmaxNewOp<CPUDevice, double>);
#endif