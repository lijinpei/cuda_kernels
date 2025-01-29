cmake_minimum_required(VERSION 3.31)

project(gpu_ops LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(Torch CONFIG REQUIRED)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/fmt)


add_subdirectory(cublas)
#add_subdirectory(cute)
