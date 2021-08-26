#include <dlfcn.h>
#include "tensorflow/core/framework/op.h" 
#include "tensorflow/core/framework/op_kernel.h" 
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("DropoutNew")
  .Input("input: T")
  .Input("random_tensor: T")
  .Attr("T: {float, double}")
  .Attr("rate: float")
#ifdef USE_SGX
  .Attr("eid_low: int")
  .Attr("eid_high: int")
#endif
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  })
  .Doc(R"doc(  
    Dropout layer.
  )doc");


template <typename Device, typename T>
class DropoutNewOp : public OpKernel {
public:
    explicit DropoutNewOp(OpKernelConstruction* context) : OpKernel(context) {
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
        const Tensor& input = context->input(0);
        const Tensor& random_tensor = context->input(1);
        const TensorShape& shape = input.shape();
        DCHECK_EQ(shape, random_tensor.shape());
            
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
        
        auto x = input.template flat<T>();
        auto random = random_tensor.template flat<T>();
#ifdef USE_SGX
        long int dim = random.size();
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float* input, float* random, float* output, long int dim, float rate);
        dlerror();

        function dropout = (function) dlsym(lib, "dropout");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of dropout failed: ", dlsym_error));
        dropout(eid_, (float*) x.data(), (float*) random.data(), (float*) output->flat<T>().data(), dim, rate_);
#else
        auto keep_mask = (random >= static_cast<T>(rate_)).template cast<T>();
        output->template flat<T>() = x*keep_mask*static_cast<T>(scale);
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
REGISTER_KERNEL_BUILDER(Name("DropoutNew").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        DropoutNewOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("DropoutNew").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        DropoutNewOp<CPUDevice, double>);
#endif