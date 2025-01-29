add_library(cublas_common_error_check error_check.cpp)
target_include_directories(cublas_common_error_check PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(cublas_common_error_check PUBLIC cublasLt)
