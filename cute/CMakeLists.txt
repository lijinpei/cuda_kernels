option(CUTLASS_INCLUDE_PATH "cutlass include path")
include_directories(${CUTLASS_INCLUDE_PATH})

add_executable(print print.cu)
target_link_libraries(print PRIVATE cuda)
