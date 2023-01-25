cmake_minimum_required (VERSION 3.2)
project(svm)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_executable(bin_to_cpp_header
    other/bin_to_cpp_header.cpp
)