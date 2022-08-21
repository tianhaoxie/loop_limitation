cmake_minimum_required(VERSION 3.10)
project(loop)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
list(APPEND CMAKE_PREFIX_PATH "/home/xth/anaconda3/envs/polygen/lib/python3.9/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

include_directories(thirdparty/libigl/include)
include_directories(thirdparty/eigen)


AUX_SOURCE_DIRECTORY(src/loopEvaluation SOURCES)
#add_executable(loop src/main.cpp ${SOURCES})
add_library(loop SHARED src/loop.cpp ${SOURCES})
target_link_libraries(loop PUBLIC ${TORCH_LIBRARIES} Python::Python )
set_property(TARGET loop PROPERTY CXX_STANDARD 14)
