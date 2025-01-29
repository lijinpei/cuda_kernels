#include "cublasLt.h"

#include "ATen/Tensor.h"
#include "ATen/core/Formatting.h"
#include "ATen/ops/allclose_native.h"
#include "ATen/ops/empty_native.h"
#include "ATen/ops/full_native.h"
#include "ATen/ops/matmul_native.h"
#include "ATen/ops/ones_native.h"
#include "ATen/ops/randn_native.h"
#include "ATen/ops/zeros_native.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/cuda/CUDAStream.h"

#include "error_check.h"

#include <cassert>
#include <cstdint>
#include <limits>

int main() {
  int M = 4096;
  int N = 4096;
  int K = 8192;
  auto gpuDev = c10::Device(c10::DeviceType::CUDA);
  auto cpuDev = c10::Device(c10::DeviceType::CPU);
  auto torchF32 = c10::ScalarType::Float;
  auto a = at::native::randn({M, K}, torchF32, std::nullopt, cpuDev);
  auto b = at::native::randn({K, N}, torchF32, std::nullopt, cpuDev);
  // auto aAcc = a.accessor<float, 2>();
  // auto bAcc = b.accessor<float, 2>();
  // for (int i = 0; i < K; ++i) {
  //   aAcc[0][i] = 1;
  //   bAcc[i][0] = 0;
  // }
  // bAcc[0][0] = 1;
  auto cRef = at::native::matmul(a, b);
  auto aGPU = a.to(gpuDev);
  auto bGPU = b.to(gpuDev);
  auto cGPU = at::native::empty_cuda({M, N}, torchF32, std::nullopt, gpuDev);
  cublasLtHandle_t handle;
  CHECK_CUBLAS(cublasLtCreate, &handle);
  cublasLtMatmulDesc_t desc;
  CHECK_CUBLAS(cublasLtMatmulDescCreate, &desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  auto ptrDev = CUBLASLT_POINTER_MODE_DEVICE;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute, desc,
               CUBLASLT_MATMUL_DESC_POINTER_MODE, &ptrDev, sizeof(ptrDev));
  cublasLtMatrixLayout_t aLayout, bLayout, cLayout;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate, &aLayout, CUDA_R_32F, M, K, K);
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate, &bLayout, CUDA_R_32F, K, N, N);
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate, &cLayout, CUDA_R_32F, M, N, N);
  auto rowMajor = CUBLASLT_ORDER_ROW;
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute, aLayout,
               CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute, bLayout,
               CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute, cLayout,
               CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor));
  cublasLtMatmulPreference_t pref;
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate, &pref);
  CHECK_CUBLAS(cublasLtMatmulPreferenceInit, pref);
  uint64_t maxWorkSpaceBytes = std::numeric_limits<uint64_t>::max();
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute, pref,
               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkSpaceBytes,
               sizeof(maxWorkSpaceBytes));
  cublasLtMatmulHeuristicResult_t heur;
  int numHeur;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic, handle, desc, aLayout, bLayout,
               cLayout, cLayout, pref, 1, &heur, &numHeur);
  CHECK_CUBLAS_RESULT(heur.state, "heuristics search result");
  cublasLtMatmulHeuristicResult_t heurRes;
  CHECK_CUBLAS(cublasLtMatmulAlgoCheck, handle, desc, aLayout, bLayout, cLayout,
               cLayout, &heur.algo, &heurRes);
  CHECK_CUBLAS_RESULT(heurRes.state, "cublaslt algo check");
  auto workspaceSize = heur.workspaceSize;
  auto workspace = at::native::empty_cuda(
      {(int64_t)workspaceSize}, c10::ScalarType::Byte, std::nullopt, gpuDev);
  auto alpha =
      at::native::full({1}, 1., std::nullopt, torchF32, std::nullopt, gpuDev);
  auto beta =
      at::native::full({1}, 0., std::nullopt, torchF32, std::nullopt, gpuDev);
  auto stream = c10::cuda::getCurrentCUDAStream();
  CHECK_CUBLAS(cublasLtMatmul, handle, desc, alpha.data_ptr(), aGPU.data_ptr(),
               aLayout, bGPU.data_ptr(), bLayout, beta.data_ptr(),
               cGPU.data_ptr(), cLayout, cGPU.data_ptr(), cLayout, &heur.algo,
               workspace.data_ptr(), workspaceSize, stream.stream());
  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy, pref);
  CHECK_CUBLAS(cublasLtMatmulDescDestroy, desc);
  CHECK_CUBLAS(cublasLtDestroy, handle);
  auto cRes = cGPU.to(cpuDev);
  double rtol = 1.3e-6;
  double atol = 1e-5;
  bool equalNAN = false;
  // at::print(std::cout, cRef, 1024);
  // std::cout << '\n';
  // at::print(std::cout, cRes, 1024);
  // std::cout << '\n';
  assert(at::native::allclose(cRef, cRes, 100 * rtol, 100 * atol, equalNAN));
}
