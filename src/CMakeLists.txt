add_library(hadamard ${CMAKE_SOURCE_DIR}/src/shared/hadamard.cu)

set_target_properties(hadamard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(winograd_cpu ${CMAKE_SOURCE_DIR}/src/winograd/winograd.cpp)

add_library(winograd_gpu ${CMAKE_SOURCE_DIR}/src/winograd/winograd.cu)

set_target_properties(winograd_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(utils ${CMAKE_SOURCE_DIR}/src/shared/utils.cpp)

add_library(loadImageGPU ${CMAKE_SOURCE_DIR}/src/shared/loadImageGPU.cu)
set_target_properties(loadImageGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(storeImageGPU ${CMAKE_SOURCE_DIR}/src/shared/storeImageGPU.cu)
set_target_properties(storeImageGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(main_winograd main_winograd.cu)
target_link_libraries(main_winograd winograd_gpu loadImageGPU storeImageGPU ${CUDA_LIBRARIES})