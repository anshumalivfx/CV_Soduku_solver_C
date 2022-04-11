cmake_minimum_required(VERSION 3.21)
project(OpenCV_Project)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)

add_executable(OpenCV_Project main.cpp Image.h Image.cpp OCR.h OCR.cpp Sudoku.h Sudoku.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV_Project ${OpenCV_LIBS})