# Créer un executable à partir de fichiers sources
add_executable(test_hadamard test_hadamard.cu)

target_link_libraries(test_hadamard hadamard ${CUDA_LIBRARIES} gtest)

add_executable(test_winograd_gpu test_winograd_gpu.cu)

target_link_libraries(test_winograd_gpu winograd_gpu ${CUDA_LIBRARIES} gtest)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_executable(test_mm_cpu test_mm_cpu.cpp)
	target_link_libraries(test_mm_cpu PUBLIC OpenMP::OpenMP_CXX gtest)

  add_executable(test_winograd_cpu test_winograd_cpu.cpp)
  target_link_libraries(test_winograd_cpu PUBLIC OpenMP::OpenMP_CXX gtest winograd_cpu)
endif()


include(GoogleTest)
gtest_discover_tests(test_hadamard)
gtest_discover_tests(test_mm_cpu)
gtest_discover_tests(test_winograd_cpu)
gtest_discover_tests(test_winograd_gpu)
