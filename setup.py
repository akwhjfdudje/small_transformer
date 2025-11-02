import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="bindings",
    ext_modules=[
        CUDAExtension(
            name="bindings",
            sources=[
                "bindings/bindings.cpp",
            ],
            include_dirs=[
                os.path.abspath("../cuda_kernels/include"),
            ],
            libraries=["cuda_kernels"],
            library_dirs=[
                os.path.abspath("../cuda_kernels/build/Release"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["--use_fast_math", "-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

