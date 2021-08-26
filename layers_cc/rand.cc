#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/random_op_cpu.h"
#include "tensorflow/core/framework/tensor_util.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("UniformDistributionNew")
    .Input("shape: int32")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {float}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return Status::OK();
    });

PHILOX_DEVICE_INLINE float Uint32ToFloat(unsigned int x) {
  const unsigned int man = x & 0x7fffffu;
  const unsigned int exp = static_cast<unsigned int>(127);
  const unsigned int val = (exp << 23) | man;

  float result;
  memcpy(&result, &val, sizeof(val));
  return result - 1.0f;
}

// Template class for your custom distribution
template <class Generator, typename RealType>
class UniformDistributionNew;

// Implementation for tf.float32
template <class Generator>
class UniformDistributionNew<Generator, float> {
 public:
  // The number of elements that will be returned (see below).
  static const int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles) (see below).
  static const int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime (see below).
  static const bool kVariableSamplesPerOutput = false;
  typedef Eigen::array<float, kResultElementCount> ResultType;
  typedef float ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      //float r = Uint32ToFloat(sample[i]);
      // Example distribution logic: produce 1 or 0 with 50% probability
      //result[i] = 1.0f * (r < 0.5f);
      result[i] = Uint32ToFloat(sample[i]);
    }
    return result;
  }
};

// Could add implementations for other data types...

// Base kernel
// Copied from core/kernels/random_op.cc
static Status AllocateOutputWithShape(OpKernelContext* ctx, const Tensor& shape,
                                      int index, Tensor** output) {
  TensorShape tensor_shape;
  TF_RETURN_IF_ERROR(tensor::MakeShape(shape, &tensor_shape));
  return ctx->allocate_output(index, tensor_shape, output);
}

template <typename Device, class Distribution>
class PhiloxRandomOp : public OpKernel {
 public:
  typedef typename Distribution::ResultElementType T;
  explicit PhiloxRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    Tensor* output;
    //OP_REQUIRES_OK(ctx, AllocateOutputWithShape(ctx, shape, 0, &output));
    OP_REQUIRES_OK(ctx, AllocateOutputWithShape(ctx, shape, 0, &output));
    auto output_flat = output->flat<T>();
    tensorflow::functor::FillPhiloxRandom<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(),
        // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
        // it just here.
        nullptr,nullptr,
        generator_.ReserveRandomOutputs(output_flat.size(), 256),
        output_flat.data(), output_flat.size(), Distribution());
  }
 private:
  GuardedPhiloxRandom generator_;
};

template struct functor::FillPhiloxRandom<CPUDevice, UniformDistributionNew<tensorflow::random::PhiloxRandom, float>>;
REGISTER_KERNEL_BUILDER(
    Name("UniformDistributionNew")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("dtype"),
    PhiloxRandomOp<CPUDevice, UniformDistributionNew<tensorflow::random::PhiloxRandom, float>>);
