cmake_minimum_required(VERSION 3.12)

# set the project name and version
project(BenchmarkCnn LANGUAGES CXX)

# set CXX standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# CUDA Libraries
find_package(CUDA)
if(CUDA_FOUND)
enable_language(CUDA)
endif()

# GoogleTest requires at least C++11
## On créée les tests pour la bibliothèque.
## Ces tests sont unitaires ou fonctionnels

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.8.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    enable_testing()

add_subdirectory(src)
add_subdirectory(test)
