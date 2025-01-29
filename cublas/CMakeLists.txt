add_subdirectory(common)

add_executable(matmul matmul.cu)
target_link_libraries(matmul PRIVATE torch cublasLt cublas_common_error_check)

add_executable(benchmark benchmark.cu)
target_link_libraries(benchmark PRIVATE torch cublasLt absl::flags absl::flags_parse absl::flags_marshalling absl::absl_log fmt::fmt cublas_common_error_check)
