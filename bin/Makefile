# gcc/5.3.0 cuda/8.0
CPP = g++
CUDA = nvcc

CXX_FLAGS = -std=c++11 -fopenmp -O1
CUDA_FLAGS = -std=c++11# -O3
CUDA_LIBS = -lcudart -I/usr/local/cuda-8.0/include -L/usr/local/cuda-8.0/lib64

vectorcpu.o: ../vectorcpu.cpp
	$(CPP) $(CXX_FLAGS) -c ../vectorcpu.cpp

cpu_compile: vectorcpu.o ../allinone.cpp ../measuring/cpu.cpp
	$(CPP) $(CXX_FLAGS) -c ../allinone.cpp
	$(CPP) $(CXX_FLAGS) -c ../measuring/cpu.cpp -o measure_cpu.o

OBJ_CPP = vectorcpu.o
OBJ_CPP_RUN = allinone.o measure_cpu.o

gpu_compile: ../csrmatrixgpu.cpp ../csrmatrixgpu.cu ../basics.cu ../allinone.cu ../allinone_gpu.cpp ../measuring/gpu.cpp
	$(CUDA) $(CUDA_FLAGS) -c ../csrmatrixgpu.cpp -o csrmatrixgpu.o
	$(CUDA) $(CUDA_FLAGS) -c ../csrmatrixgpu.cu  -o csrmatrixgpu_c.o
	$(CUDA) $(CUDA_FLAGS) -c ../basics.cu        -o basics_c.o
	$(CUDA) $(CUDA_FLAGS) -c ../allinone.cu      -o allinone_c.o
	$(CPP) $(CXX_FLAGS) $(CUDA_LIBS) -c ../allinone_gpu.cpp  -o allinone_gpu.o
	$(CPP) $(CXX_FLAGS) $(CUDA_LIBS) -c ../measuring/gpu.cpp -o measure_gpu.o

OBJ_CUDA = csrmatrixgpu.o csrmatrixgpu_c.o basics_c.o allinone_c.o
#OBJ_CUDA = csrmatrixgpu.o csrmatrixgpu.co basics.co allinone.co
OBJ_CUDA_RUN = allinone_gpu.o measure_gpu.o

cpu_run: cpu_compile
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) allinone.o -o run_cpu
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) measure_cpu.o -o measure_cpu

gpu_run: cpu_compile gpu_compile
	$(CPP) $(CXX_FLAGS) $(CUDA_LIBS) $(OBJ_CUDA) $(OBJ_CPP) allinone_gpu.o -o run_cuda
	$(CPP) $(CXX_FLAGS) $(CUDA_LIBS) $(OBJ_CUDA) $(OBJ_CPP) measure_gpu.o -o measure_gpu

all: cpu_run gpu_run

clean:
	rm -rf $(OBJ_CPP)  $(OBJ_CPP_RUN)
	rm -rf $(OBJ_CUDA) $(OBJ_CUDA_RUN)

cleanfull: clean
	rm -rf run_cpu  measure_cpu
	rm -rf run_cuda measure_gpu
