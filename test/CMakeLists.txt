# Specify the test source files
set(TEST_SOURCES test_hadamard.cu)

# Specify the shared code to link with the test
set(SHARED_SOURCES ${CMAKE_SOURCE_DIR}/src/shared/hadamard.cu)

# Create the test executable
add_executable(test_hadamard ${TEST_SOURCES} ${SHARED_SOURCES})

# Link the GoogleTest libraries
target_link_libraries(test_hadamard GTest::GTest GTest::Main)

# Include directories
target_include_directories(test_hadamard PRIVATE ${CMAKE_SOURCE_DIR}/src/shared)

# Enable CUDA compilation if needed
find_package(CUDA REQUIRED)
target_link_libraries(test_hadamard ${CUDA_LIBRARIES})

# Add test functionality
include_directories(${GTEST_INCLUDE_DIRS})
add_test(NAME test_hadamard COMMAND test_hadamard)
