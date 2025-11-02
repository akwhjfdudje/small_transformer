#include <torch/extension.h>
#include "linalg/linalg.cuh"
#include "activate/activate.cuh"
#include "arith/arith.cuh"

torch::Tensor relu_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda());
    auto output = torch::empty_like(input);
    vectorReLU(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    return output;
}

torch::Tensor matmul_binding(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda());
    int N = A.size(0);
    auto C = torch::empty({N, B.size(1)}, A.options());
    matrixMul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

torch::Tensor softmax_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda());
    auto output = torch::empty_like(input);
    softmax(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &relu_binding, "ReLU (CUDA)");
    m.def("matmul", &matmul_binding, "Matrix Multiply (CUDA)");
    m.def("softmax", &softmax_binding, "Softmax (CUDA)");
}
