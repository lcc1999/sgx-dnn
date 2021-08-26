#include <dlfcn.h>
#include "tensorflow/core/framework/op.h" 
#include "tensorflow/core/framework/op_kernel.h" 
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


REGISTER_OP("DenseNew")
  .Attr("T: {float, double}")
  .Attr("use_bias: bool=true")
#ifdef USE_SGX
  .Attr("eid_low: int")
  .Attr("eid_high: int")
#endif
  .Input("input: T")
  .Input("weights: T")
  .Input("bias: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

    shape_inference::ShapeHandle weight_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
    
    shape_inference::ShapeHandle bias_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bias_shape));
  
    shape_inference::DimensionHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(input_shape, 1), c->Dim(weight_shape, 0), &merged));
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(weight_shape, 1), c->Dim(bias_shape, 0), &merged));

    c->set_output(0, c->Matrix(c->Dim(input_shape, 0), c->Dim(weight_shape, 1)));
    return Status::OK();
  })
  .Doc(R"doc(  
    Dense layer.
  )doc");


template <typename Device, typename T>
class DenseNewOp : public OpKernel {
public:
    explicit DenseNewOp(OpKernelConstruction* context) : OpKernel(context) {
#ifdef USE_SGX
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("layers_sgx.so", RTLD_NOW);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load layers_sgx.so!"));
#endif
        OP_REQUIRES_OK(context, context->GetAttr("use_bias", &use_bias_));
    }

    void Compute(OpKernelContext* context) override {
        DCHECK_EQ(context->num_inputs(),3);
        const Tensor& input = context->input(0);
        const Tensor& weights = context->input(1);
        const Tensor& bias = context->input(2);
        
        const TensorShape& input_shape = input.shape();
        const TensorShape& weights_shape = weights.shape();
        const TensorShape& bias_shape = bias.shape();
        
        DCHECK_EQ(input_shape.dims(), 2);
        DCHECK_EQ(weights_shape.dims(), 2);
        DCHECK_EQ(bias_shape.dims(), 1);
        DCHECK_EQ(weights_shape.dim_size(0),input_shape.dim_size(1));
        DCHECK_EQ(bias_shape.dim_size(0),weights_shape.dim_size(1));
    
        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(weights_shape.dim_size(1));
            
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
        /*auto input_tensor = input.matrix<T>();
        auto weights_tensor = weights.matrix<T>();
        auto bias_tensor = bias.flat<T>();
        auto output_tensor = output->matrix<T>();
    
        for (int i = 0; i < input_shape.dim_size(0); i++) {
            for (int j = 0; j < weights_shape.dim_size(1); j++) {
                output_tensor(i, j) = 0;
                for(int k = 0; k < weights_shape.dim_size(0); k++)
                    output_tensor(i, j) += input_tensor(i, k) * weights_tensor(k, j);
                if(use_bias_)
                    output_tensor(i, j) += bias_tensor(j);
            }
        }*/

#ifdef USE_SGX
        long int dim_in[2] = {input_shape.dim_size(0), input_shape.dim_size(1)};
        long int dim_w[2] = {weights_shape.dim_size(0), weights_shape.dim_size(1)};
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float* input, float* output, float* weights, float* bias,
								  	  long int dim_in[2], long int dim_w[2]);
        dlerror();

        function dense = (function) dlsym(lib, "dense");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of dense failed: ", dlsym_error));
        dense(eid_, (float*) input.flat<T>().data(), (float*) output->flat<T>().data(), 
            (float*) weights.flat<T>().data(), (float*) bias.flat<T>().data(),dim_in, dim_w);
#else
        auto x=input.tensor<T, 2>();
        auto w=weights.tensor<T, 2>();
        auto b=bias.tensor<T, 1>();
        Eigen::array<int64, 2> two_dims{{1, bias_shape.dim_size(0)}};
        Eigen::array<int64, 2> bcast({input_shape.dim_size(0), 1});
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
        output->tensor<T, 2>() = x.contract(w,product_dims) + b.reshape(two_dims).broadcast(bcast);
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
REGISTER_KERNEL_BUILDER(Name("DenseNew").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        DenseNewOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("DenseNew").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        DenseNewOp<CPUDevice, double>);
#endif