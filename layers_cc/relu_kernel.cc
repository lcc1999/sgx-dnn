#include <dlfcn.h>
#include "tensorflow/core/framework/op.h" 
#include "tensorflow/core/framework/op_kernel.h" 
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("ReluNew")
  .Attr("T: {float, double}")
#ifdef USE_SGX
  .Attr("eid_low: int")
  .Attr("eid_high: int")
#endif
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  })
  .Doc(R"doc(  
    Relu layer.
  )doc");


template <typename Device, typename T>
class ReluNewOp : public OpKernel {
public:
    explicit ReluNewOp(OpKernelConstruction* context) : OpKernel(context) {
#ifdef USE_SGX
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("layers_sgx.so", RTLD_NOW);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load layers_sgx.so!"));
#endif
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const TensorShape& shape = input.shape();
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
        
        auto x = input.template flat<T>();
#ifdef USE_SGX
        long int dim = x.size();
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float* input, float* output, long int dim);
        dlerror();

        function relu = (function) dlsym(lib, "relu");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of relu failed: ", dlsym_error));
        relu(eid_, (float*) x.data(), (float*) output->flat<T>().data(), dim);
#else
        output->template flat<T>() = x.template cwiseMax<Eigen::PropagateNaN>(static_cast<T>(0));
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
REGISTER_KERNEL_BUILDER(Name("ReluNew").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        ReluNewOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("ReluNew").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        ReluNewOp<CPUDevice, double>);
#endif