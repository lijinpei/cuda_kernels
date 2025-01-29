#include "cuda.h"
#include "cute/util/print.hpp"

__global__ void print_tid() { cute::print("%d\n", threadIdx.x); }

int main() {
  print_tid<<<4, 128>>>();
  cuStreamSynchronize(nullptr);
}
