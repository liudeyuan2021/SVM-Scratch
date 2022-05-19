cmake_minimum_required (VERSION 3.2)
project(svm)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_executable(file_to_mem
    other/file_to_mem.cpp
)