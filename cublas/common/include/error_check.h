#pragma once

#include "cublasLt.h"

#define CHECK_CUBLAS(FUNC, ...)                                                \
  check_cublasLt_impl(FUNC(__VA_ARGS__), #FUNC, __FILE__, __LINE__)
#define CHECK_CUBLAS_RESULT(RESULT, DESC)                                      \
  check_cublasLt_impl(RESULT, DESC, __FILE__, __LINE__)

void check_cublasLt_impl(cublasStatus_t status, const char *func,
                         const char *file, int line);
