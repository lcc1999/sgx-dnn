#include "Enclave.h"
#include <unsupported/Eigen/CXX11/Tensor>


void ecall_dense(float* input, float* output, float* weights, float* bias, 
        long int dim_in[2], long int dim_w[2])
{
    /*Eigen::Tensor<float, 1, Eigen::RowMajor> b = Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>>(bias, dim_w[1]);
    Eigen::Tensor<float, 2, Eigen::RowMajor> x = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(input, dim_in[0], dim_in[1]);
    Eigen::Tensor<float, 2, Eigen::RowMajor> w = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(weights, dim_w[0], dim_w[1]);
    Eigen::Tensor<float, 2, Eigen::RowMajor> y = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(output, dim_in[0], dim_w[1]);*/
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> x(input, dim_in[0], dim_in[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> w(weights, dim_w[0], dim_w[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> b(bias, dim_w[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> y(output, dim_in[0], dim_w[1]);
    /*TensorMap<float, 2> x(input, dim_in[0], dim_in[1]);
    TensorMap<float, 2> w(weights, dim_w[0], dim_w[1]);
	TensorMap<float, 1> b(bias, dim_w[1]);
    TensorMap<float, 2> y(output, dim_in[0], dim_w[1]);*/
    Eigen::array<int, 2> two_dims{{1, dim_w[1]}};
    Eigen::array<int, 2> bcast({dim_in[0], 1});
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    y = x.contract(w,product_dims) + b.reshape(two_dims).broadcast(bcast);
}

void ecall_dense_backward(float* input, float* weights, float* grad, float* grad_input, 
        float* grad_weights, float* grad_bias, long int dim_in[2], long int dim_w[2])
{   /*Eigen::Tensor<float, 2, Eigen::RowMajor> b = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(input, dim_in[0], dim_in[1]);
    Eigen::Tensor<float, 2, Eigen::RowMajor> w = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(weights, dim_w[0], dim_w[1]);
    Eigen::Tensor<float, 2, Eigen::RowMajor> g = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(grad, dim_in[0], dim_w[1]);
    Eigen::Tensor<float, 2, Eigen::RowMajor> w_g = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(grad_weights, dim_w[0], dim_w[1]);
	Eigen::Tensor<float, 1, Eigen::RowMajor> b_g = Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>>(grad_bias, dim_w[1]);
    Eigen::Tensor<float, 2, Eigen::RowMajor> x_g = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>(grad_input, dim_in[0], dim_in[1]);*/
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> x(input, dim_in[0], dim_in[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> w(weights, dim_w[0], dim_w[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> g(grad, dim_in[0], dim_w[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> w_g(grad_weights, dim_w[0], dim_w[1]);
	Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> b_g(grad_bias, dim_w[1]);
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> x_g(grad_input, dim_in[0], dim_in[1]);
    /*TensorMap<float, 2> x(input, dim_in[0], dim_in[1]);
    TensorMap<float, 2> w(weights, dim_w[0], dim_w[1]);
    TensorMap<float, 2> g(grad, dim_in[0], dim_w[1]);
    TensorMap<float, 2> w_g(grad_weights, dim_w[0], dim_w[1]);
	TensorMap<float, 1> b_g(grad_bias, dim_w[1]);
    TensorMap<float, 2> x_g(grad_input, dim_in[0], dim_in[1]);*/
    
    
    Eigen::array<Eigen::IndexPair<int>, 1> gx_dim = { Eigen::IndexPair<int>(1, 1) };
    Eigen::array<Eigen::IndexPair<int>, 1> gw_dim = { Eigen::IndexPair<int>(0, 0) };
    Eigen::array<int, 1> gb_dims({0});
    x_g = g.contract(w,gx_dim);
    w_g = x.contract(g,gw_dim);
    b_g = g.sum(gb_dims);
}
