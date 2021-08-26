#include "Enclave.h"
#include <unsupported/Eigen/CXX11/Tensor>


void ecall_softmax(float* input, float* output, long int dim[2])
{
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> x(input, dim[0], dim[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> y(output, dim[0], dim[1]);

    Eigen::DSizes<int, 1> along_class(1);
    Eigen::DSizes<int, 2> batch_by_one(dim[0], 1);
    Eigen::DSizes<int, 2> one_by_class(1, dim[1]);
    auto shifted_logits = (x - x.maximum(along_class).eval()
                                                     .reshape(batch_by_one)
                                                     .broadcast(one_by_class)).exp();
    y = (shifted_logits * shifted_logits.sum(along_class)
                                        .inverse()
                                        .eval()
                                        .reshape(batch_by_one)
                                        .broadcast(one_by_class));
}

void ecall_softmax_backward(float* softmax, float* grad, float* grad_input, long int dim[2])
{   
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> y(softmax, dim[0], dim[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> g(grad, dim[0], dim[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> g_x(grad_input, dim[0], dim[1]);
        
    Eigen::DSizes<int, 1> along_class(1);
    Eigen::DSizes<int, 2> batch_by_one(dim[0], 1);
    Eigen::DSizes<int, 2> one_by_class(1, dim[1]);
    g_x = y * (g - (y * g).sum(along_class)
                          .eval().reshape(batch_by_one)
                          .broadcast(one_by_class));
}
