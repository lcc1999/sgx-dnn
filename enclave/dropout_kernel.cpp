#include "Enclave.h"
#include <unsupported/Eigen/CXX11/Tensor>


void ecall_dropout(float* input, float* random, float* output, long int dim, float rate)
{
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> x(input, dim);
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> r(random, dim);
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> y(output, dim);
    
    float scale = 1 / (1-rate);
    auto keep_mask = (r >= static_cast<float>(rate)).cast<float>();
    y = x*keep_mask*static_cast<float>(scale);
}

void ecall_dropout_backward(float* random, float* grad, float* grad_input, long int dim, float rate)
{   
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> r(random, dim);
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> g(grad, dim);
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> x_g(grad_input, dim);
    
    float scale = 1 / (1-rate);
    auto keep_mask = (r >= static_cast<float>(rate)).cast<float>();
    x_g = g*keep_mask*static_cast<float>(scale);
}
