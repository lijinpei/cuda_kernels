#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "fmt/format.h"
#include "fmt/ostream.h"

#include "error_check.h"

#include <iostream>

ABSL_FLAG(int, B, 1, "GEMM problem batch size B");
ABSL_FLAG(int, M, 512, "GEMM problem shape M");
ABSL_FLAG(int, N, 512, "GEMM problem shape N");
ABSL_FLAG(int, K, 512, "GEMM problem shape K");

int main(int argc, char *argv[]) {
  auto unknown_args = absl::ParseCommandLine(argc, argv);
  auto num_unknown_args = (int)unknown_args.size() > 1;
  if (num_unknown_args > 0) {
    fmt::print(std::cerr, "met {} unknown command line arguments\n",
               num_unknown_args);
  }
}
