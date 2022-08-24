cmake_minimum_required(VERSION 3.10)
project(LPBF)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(main main.cpp hog.h hog.cpp svm.h svm.cpp)
target_link_libraries(main ${OpenCV_LIBS})