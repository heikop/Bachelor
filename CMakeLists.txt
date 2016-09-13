##################################################
###                                            ###
###                  GENERAL                   ###
###                                            ###
##################################################

cmake_minimum_required (VERSION 2.8)
project (BACHELOR)


##################################################
###                                            ###
###                  PACKAGES                  ###
###                                            ###
##################################################

find_package (MPI REQUIRED)
find_package (CUDA REQUIRED)

include_directories (/opt/openmpi/include)
include(FindCUDA)

##################################################
###                                            ###
###                    FLAGS                   ###
###                                            ###
##################################################

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")

#list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
LIST(APPEND CUDA_NVCC_FLAGS "--compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg")
LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20")
LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

##################################################
###                                            ###
###                 EXECUTABLES                ###
###                                            ###
##################################################

file (GLOB GLOBSOURCES "*.cpp")
add_library (GLOBLIB ${GLOBSOURCES})
file (GLOB CUDASOURCES "*.cu")
cuda_add_library(CUDALIB ${CUDASOURCES})

cuda_add_executable(demo "main.cpp")
target_link_libraries(demo GLOBLIB CUDALIB ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
