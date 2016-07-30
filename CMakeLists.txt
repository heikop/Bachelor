##################################################
###                                            ###
###                  GENERAL                   ###
###                                            ###
##################################################

cmake_minimum_required (VERSION 2.8)
project (BACHELOR)
# enable testing also in the build dir (not only in test subdir)
include(CTest) # this includes enable_testing() as well


##################################################
###                                            ###
###                  PACKAGES                  ###
###                                            ###
##################################################

# TODO add doxygen(?) and OpenCL or CUDA(?)
find_package (MPI REQUIRED)
find_package (CUDA REQUIRED)

include_directories (/opt/openmpi/include)
include(FindCUDA)


##################################################
###                                            ###
###               SUBDIRECTORIES               ###
###                                            ###
##################################################

#add_subdirectory (src)
#add_subdirectory (test)
#add_subdirectory (demos)


##################################################
###                                            ###
###               DOCUMENTATION                ###
###                                            ###
##################################################

# TODO (later)




#file (GLOB GLOBSOURCES "*.cpp" "*.cu")
file (GLOB GLOBSOURCES "*.cpp")
add_library (GLOBLIB ${GLOBSOURCES})
file (GLOB CUDASOURCES "*.cu")
cuda_add_library(CUDALIB ${CUDASOURCES})


#file (GLOB DEMO "main.cpp")
#add_executable(demo "main.cpp")

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_executable(demo "main.cpp")
target_link_libraries(demo GLOBLIB CUDALIB ${MPI_LIBRARIES} ${CUDA_LIBRARIES})






#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Icpp11-range --std=c++11)
#LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
#LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler "std=c++0x")
#LIST(APPEND CUDA_NVCC_FLAGS -Icpp11-range)
#LIST(APPEND CUDA_NVCC_FLAGS --std=c++11)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
