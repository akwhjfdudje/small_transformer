cmake_minimum_required(VERSION 3.18)
project(transformer_bindings LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)

add_library(bindings MODULE bindings/bindings.cpp)
set_target_properties(bindings PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(bindings PRIVATE
    ../cuda_kernels/include
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(bindings PRIVATE
    ../../cuda_kernels/build/Release/cuda_kernels
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

set_target_properties(bindings PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
