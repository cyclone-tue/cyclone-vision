cmake_minimum_required(VERSION 3.8)
project(Vision)

SET(CMAKE_MODULE_PATH /usr/local/lib/cmake)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)
include_directories(/usr/local/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

macro( config_project PROJNAME LIBNAME )
    include_directories( ${${LIBNAME}_INCLUDE_DIR} )
    target_link_libraries( ${PROJNAME} ${${LIBNAME}_LIBRARY} )
endmacro()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_executable(Vision main.cpp)
file(COPY red2.txt green2.txt angles2.txt DESTINATION ./)

target_link_libraries(Vision
        ${OpenCV_LIBS}
        )