add_library(hadamard ${CMAKE_SOURCE_DIR}/src/shared/hadamard.cu)

set_target_properties(hadamard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(winograd_cpu ${CMAKE_SOURCE_DIR}/src/winograd/winograd.cpp)

add_library(winograd_gpu ${CMAKE_SOURCE_DIR}/src/winograd/winograd.cu)

set_target_properties(winograd_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(utils ${CMAKE_SOURCE_DIR}/src/shared/utils.cpp)
