add_library(hadamard ${CMAKE_SOURCE_DIR}/src/shared/hadamard.cu)

set_target_properties(hadamard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
