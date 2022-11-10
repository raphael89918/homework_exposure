cmake_minimum_required(VERSION 3.10)
project(autohsv)

# gcc
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CX11_ABI=0")

# OPENCV
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#REALSENSE
find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()

add_executable(main main.cpp)

add_library(tesseract_OCR tesseract_OCR.cpp)
# add_library(autothreshold autothreshold.cpp)

#target_link_libraries
# target_link_libraries(autohsv autothreshold)
target_link_libraries(main ${OpenCV_LIBS} )
target_link_libraries(main libtesseract)
target_link_libraries(main leptonica)
target_link_libraries(main tesseract_OCR)
target_link_libraries(main ${REALSENSE2_FOUND})
