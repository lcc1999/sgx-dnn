#include "Enclave.h"
#include <unsupported/Eigen/CXX11/Tensor>


void ecall_relu(float* input, float* output, long int dim)
{
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> x(input, dim);
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> y(output, dim);

    y = x.cwiseMax(static_cast<float>(0));
}

void ecall_relu_backward(float* input, float* grad, float* grad_input, long int dim)
{   
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> x(input, dim);
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> g(grad, dim);
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> g_x(grad_input, dim);
        
    g_x = g * (x > static_cast<float>(0)).cast<float>();
}
