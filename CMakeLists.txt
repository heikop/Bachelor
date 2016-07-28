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




file (GLOB GLOBSOURCES "*.cpp" "*.cu")
add_library (GLOBLIB ${GLOBSOURCES})


#file (GLOB DEMO "main.cpp")
add_executable(demo "main.cpp")
target_link_libraries(demo GLOBLIB ${MPI_LIBRARIES})







LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)





