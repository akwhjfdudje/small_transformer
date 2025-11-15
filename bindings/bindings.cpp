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
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be 3D or 4D");
    TORCH_CHECK(B.dim() == A.dim(), "A and B must have same rank");

    // Shapes:
    // A: (batch, M, K)
    // B: (batch, K, N)
    // or if 4D:
    // A: (batch, heads, M, K)
    // B: (batch, heads, K, N)

    bool is4D = (A.dim() == 4);

    int batch = is4D ? A.size(0) * A.size(1) : A.size(0);
    int M     = A.size(A.dim() - 2);
    int K     = A.size(A.dim() - 1);
    int KB    = B.size(B.dim() - 2); // must equal K
    int N     = B.size(B.dim() - 1);

    TORCH_CHECK(K == KB, "Inner dimensions must match (A[*,*,", K, "] vs B[*,", KB, ",*])");

    // Flatten batch dimension if 4D:
    auto A_flat = A.contiguous().view({batch, M, K});
    auto B_flat = B.contiguous().view({batch, K, N});

    auto C = torch::empty({batch, M, N}, A.options());

    // Call updated CUDA kernel
    batchedMatrixMul(
        A_flat.data_ptr<float>(),
        B_flat.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N, batch
    );

    // If original was 4D, restore shape:
    if (is4D) {
        int outerBatch = A.size(0);
        int heads      = A.size(1);
        C = C.view({outerBatch, heads, M, N});
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
