# gcc/5.3.0 cuda/8.0
CPP = g++
CUDA = nvcc

CXX_FLAGS = -std=c++11 -fopenmp -O1
CUDA_FLAGS = -std=c++11# -O3
CUDA_LIBS = -lcudart -I/usr/local/cuda-8.0/include -L/usr/local/cuda-8.0/lib64

vectorcpu.o: ../vectorcpu.cpp
	$(CPP) $(CXX_FLAGS) -c ../vectorcpu.cpp

cpp: vectorcpu.o ../allinone.cpp
	$(CPP) $(CXX_FLAGS) -c ../allinone.cpp

OBJ_CPP = vectorcpu.o
OBJ_CPP_RUN = allinone.o

cuda: ../csrmatrixgpu.cpp ../csrmatrixgpu.cu ../basics.cu ../allinone.cu ../allinone_gpu.cpp
	$(CUDA) $(CUDA_FLAGS) -c ../csrmatrixgpu.cpp -o csrmatrixgpu.o
	$(CUDA) $(CUDA_FLAGS) -c ../csrmatrixgpu.cu  -o csrmatrixgpu_c.o
	$(CUDA) $(CUDA_FLAGS) -c ../basics.cu        -o basics_c.o
	$(CUDA) $(CUDA_FLAGS) -c ../allinone.cu      -o allinone_c.o
	$(CPP) $(CXX_FLAGS) $(CUDA_LIBS) -c ../allinone_gpu.cpp -o allinone_gpu.o

OBJ_CUDA = csrmatrixgpu.o csrmatrixgpu_c.o basics_c.o allinone_c.o
#OBJ_CUDA = csrmatrixgpu.o csrmatrixgpu.co basics.co allinone.co
OBJ_CUDA_RUN = allinone_gpu.o

all: cpp cuda
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) allinone.o -o run_cpp
	$(CPP) $(CXX_FLAGS) $(CUDA_LIBS) $(OBJ_CUDA) $(OBJ_CPP) allinone_gpu.o -o run_cuda

clean:
	rm -rf run_cpp $(OBJ_CPP) $(OBJ_CPP_RUN)
	rm -rf run_cuda $(OBJ_CUDA) $(OBJ_CUDA_RUN)
