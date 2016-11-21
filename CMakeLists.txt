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

#find_package (MPI REQUIRED)
#find_package (CUDA REQUIRED)

#include_directories (/opt/openmpi/include)
#include(FindCUDA)

##################################################
###                                            ###
###                    FLAGS                   ###
###                                            ###
##################################################

#list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -ffast-math -Wall")
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -O3 -ffast-math -Wall")
#list(APPEND CMAKE_CXX_FLAGS "-fopenmp")

#list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
#LIST(APPEND CUDA_NVCC_FLAGS "--compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg")
#LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")
#list(APPEND CUDA_NVCC_FLAGS "-O3")
#LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20")
#LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
#LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
#
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(OpenMP)
#if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    #list (APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #list (APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()
#list(APPEND CMAKE_CXX_FLAGS "-DNDEBUG")
#list(APPEND CMAKE_C_FLAGS "-DNDEBUG")

##################################################
###                                            ###
###                 EXECUTABLES                ###
###                                            ###
##################################################

file (GLOB GLOBSOURCES "*.cpp")
add_library (GLOBLIB ${GLOBSOURCES})
#file (GLOB CUDASOURCES "*.cu")
#cuda_add_library(CUDALIB ${CUDASOURCES})

add_executable(demo "main.cpp")
target_link_libraries(demo GLOBLIB    ) # CUDALIB ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
add_executable(convert "converter.cpp")
target_link_libraries(convert GLOBLIB ) # CUDALIB ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
add_executable(example1 "example1.cpp")
target_link_libraries(example1 GLOBLIB) # CUDALIB ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
add_executable(example "example.cpp"  )
target_link_libraries(example GLOBLIB ) # CUDALIB ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
add_executable(allinone "allinone.cpp")
target_link_libraries(allinone GLOBLIB) # CUDALIB ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
