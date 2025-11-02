cmake_minimum_required(VERSION 3.18)
project(bindings LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MD")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

option(TORCH_PATH "Path to LibTorch installation" "")

if (NOT TORCH_PATH)
    message(FATAL_ERROR "Please specify LibTorch path with -DTORCH_PATH=/path/to/libtorch")
endif()

set(Torch_DIR "${TORCH_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED)

add_library(bindings MODULE bindings/bindings.cpp)

target_link_libraries(bindings PRIVATE
    ../../cuda_kernels/build/Release/cuda_kernels
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

target_include_directories(bindings PRIVATE
    ../cuda_kernels/include
    ${Python3_INCLUDE_DIRS}
)

set_target_properties(bindings PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    PREFIX ""              
    SUFFIX ".pyd"
)
