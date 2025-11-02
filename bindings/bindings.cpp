#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include "linalg/linalg.cuh"
#include "activate/activate.cuh"
#include "arith/arith.cuh"

torch::Tensor relu_binding(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda());
    auto output = torch::empty_like(input);
    vectorReLU(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    return output;
}

torch::Tensor batched_matmul_binding(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda());
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be 3D or 4D for batched matmul");
    
    int batch = (A.dim() == 4) ? A.size(0)*A.size(1) : A.size(0); // flatten batch & heads if 4D
    int N = A.size(A.dim()-2);   // rows
    int M = B.size(B.dim()-1);   // cols

    // flatten last two dims for kernel
    auto A_flat = A.contiguous().view({batch, N, A.size(A.dim()-1)});
    auto B_flat = B.contiguous().view({batch, B.size(B.dim()-2), M});

    auto C = torch::empty({batch, N, M}, A.options());

    batchedMatrixMul(A_flat.data_ptr<float>(), B_flat.data_ptr<float>(), C.data_ptr<float>(), N, batch);

    // reshape back if original was 4D
    if (A.dim() == 4) {
        int heads = A.size(1);
        int T = A.size(2);
        C = C.view({A.size(0), heads, T, T});
    }

    return C;
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
    m.def("batched_matmul", &batched_matmul_binding, "Batched Matrix Multiply (CUDA)");
    m.def("relu", &relu_binding, "ReLU (CUDA)");
    m.def("matmul", &matmul_binding, "Matrix Multiply (CUDA)");
    m.def("softmax", &softmax_binding, "Softmax (CUDA)");
}
