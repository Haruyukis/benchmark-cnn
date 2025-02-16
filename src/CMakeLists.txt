add_library(hadamard ${CMAKE_SOURCE_DIR}/src/shared/hadamard.cu)

set_target_properties(hadamard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(winograd_cpu ${CMAKE_SOURCE_DIR}/src/winograd/winograd.cpp)

add_library(winograd_gpu ${CMAKE_SOURCE_DIR}/src/winograd/winograd.cu)

set_target_properties(winograd_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(utils ${CMAKE_SOURCE_DIR}/src/shared/utils.cpp)
enable_language(C)
add_library(storeImageCPU ${CMAKE_SOURCE_DIR}/src/shared/storeImage.c)
add_library(loadImageCPU ${CMAKE_SOURCE_DIR}/src/shared/loadImage.c)
set_target_properties(loadImageCPU PROPERTIES LINKER_LANGUAGE C)
set_target_properties(storeImageCPU PROPERTIES LINKER_LANGUAGE C)

add_library(loadImageGPU ${CMAKE_SOURCE_DIR}/src/shared/loadImageGPU.cu)
set_target_properties(loadImageGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(storeImageGPU ${CMAKE_SOURCE_DIR}/src/shared/storeImageGPU.cu)
set_target_properties(storeImageGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(main_winograd_gpu main_winograd_gpu.cu)
target_link_libraries(main_winograd_gpu winograd_gpu loadImageGPU storeImageGPU utils ${CUDA_LIBRARIES})

add_executable(main_winograd_cpu main_winograd_cpu.cpp)
target_link_libraries(main_winograd_cpu winograd_cpu loadImageCPU storeImageCPU utils)