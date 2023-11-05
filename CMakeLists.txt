project(AutoStitching)

cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

find_package(OpenMP)

if(OPENMP_FOUND)
	option(WITH_OPENMP ON)
	message(STATUS "With OpenMP ")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAX_OPENMP_NUM_THREADS=${PROCESSOR_COUNT} -DOPENMP_NUM_THREADS=${PROCESSOR_COUNT} ${OpenMP_CXX_FLAGS} -DOPENMP")
endif(OPENMP_FOUND)

find_package(LAPACK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(BLAS REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/../argparse/ ${CMAKE_SOURCE_DIR}/../levmar-2.6/)
include_directories(${CMAKE_SOURCE_DIR} ${OPENCV_INCLUDE_DIR} ${LAPACK_INCLUDE_DIRS} ${BLAS_INCLUDE_DIRS} )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_subdirectory(${CMAKE_SOURCE_DIR}/src)