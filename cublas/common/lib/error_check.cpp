#include "error_check.h"

#include <cassert>
#include <iostream>

void check_cublasLt_impl(cublasStatus_t status, const char *func,
                         const char *file, int line) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return;
  }
  std::cerr << "cublas " << func << " failed on file " << file << " +" << line
            << " : error name: " << cublasLtGetStatusName(status)
            << " error message: " << cublasLtGetStatusString(status) << '\n';
  assert(false);
}
