# Créer un executable à partir de fichiers sources
add_executable(test_hadamard test_hadamard.cu)

target_link_libraries(test_hadamard hadamard ${CUDA_LIBRARIES} gtest)

include(GoogleTest)
gtest_discover_tests(test_hadamard)
