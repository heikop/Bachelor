# gcc/5.3.0 cuda/8.0
CPP = g++
CUDA = nvcc

CXX_FLAGS = -std=c++11 -fopenmp -O1
CUDA_FLAGS = -std=c++11# -O3
CUDA_LIBS = -lcudart -I/usr/local/cuda-8.0/include -L/usr/local/cuda-8.0/lib64

vectorcpu.o: ../../../vectorcpu.cpp
	$(CPP) $(CXX_FLAGS) -c ../../../vectorcpu.cpp

cpu_compile: vectorcpu.o \
			 ../cpu_csr_q1_glob_loop_nopre_f_000.cpp \
			 ../cpu_csr_q1_glob_loop_nopre_d_001.cpp \
			  ../cpu_csr_q1_loc_loop_nopre_f_008.cpp \
			  ../cpu_csr_q1_loc_loop_nopre_d_009.cpp \
			 ../cpu_csr_q2_glob_loop_nopre_f_016.cpp \
			 ../cpu_csr_q2_glob_loop_nopre_d_017.cpp \
			  ../cpu_csr_q2_loc_loop_nopre_f_024.cpp \
			  ../cpu_csr_q2_loc_loop_nopre_d_025.cpp \
			 ../cpu_lma_q1_glob_loop_nopre_f_032.cpp \
			 ../cpu_lma_q1_glob_loop_nopre_d_033.cpp \
			  ../cpu_lma_q1_loc_loop_nopre_f_040.cpp \
			  ../cpu_lma_q1_loc_loop_nopre_d_041.cpp \
			 ../cpu_lma_q2_glob_loop_nopre_f_048.cpp \
			 ../cpu_lma_q2_glob_loop_nopre_d_049.cpp \
			  ../cpu_lma_q2_loc_loop_nopre_f_056.cpp \
			  ../cpu_lma_q2_loc_loop_nopre_d_057.cpp
	$(CPP) $(CXX_FLAGS) -c ../cpu_csr_q1_glob_loop_nopre_f_000.cpp -o cpu_csr_q1_glob_loop_nopre_f_000.o
	$(CPP) $(CXX_FLAGS) -c ../cpu_csr_q1_glob_loop_nopre_d_001.cpp -o cpu_csr_q1_glob_loop_nopre_d_001.o
	$(CPP) $(CXX_FLAGS) -c  ../cpu_csr_q1_loc_loop_nopre_f_008.cpp -o  cpu_csr_q1_loc_loop_nopre_f_008.o
	$(CPP) $(CXX_FLAGS) -c  ../cpu_csr_q1_loc_loop_nopre_d_009.cpp -o  cpu_csr_q1_loc_loop_nopre_d_009.o
	$(CPP) $(CXX_FLAGS) -c ../cpu_csr_q2_glob_loop_nopre_f_016.cpp -o cpu_csr_q2_glob_loop_nopre_f_016.o
	$(CPP) $(CXX_FLAGS) -c ../cpu_csr_q2_glob_loop_nopre_d_017.cpp -o cpu_csr_q2_glob_loop_nopre_d_017.o
	$(CPP) $(CXX_FLAGS) -c  ../cpu_csr_q2_loc_loop_nopre_f_024.cpp -o  cpu_csr_q2_loc_loop_nopre_f_024.o
	$(CPP) $(CXX_FLAGS) -c  ../cpu_csr_q2_loc_loop_nopre_d_025.cpp -o  cpu_csr_q2_loc_loop_nopre_d_025.o
	$(CPP) $(CXX_FLAGS) -c ../cpu_lma_q1_glob_loop_nopre_f_032.cpp -o cpu_lma_q1_glob_loop_nopre_f_032.o
	$(CPP) $(CXX_FLAGS) -c ../cpu_lma_q1_glob_loop_nopre_d_033.cpp -o cpu_lma_q1_glob_loop_nopre_d_033.o
	$(CPP) $(CXX_FLAGS) -c  ../cpu_lma_q1_loc_loop_nopre_f_040.cpp -o  cpu_lma_q1_loc_loop_nopre_f_040.o
	$(CPP) $(CXX_FLAGS) -c  ../cpu_lma_q1_loc_loop_nopre_d_041.cpp -o  cpu_lma_q1_loc_loop_nopre_d_041.o
	$(CPP) $(CXX_FLAGS) -c ../cpu_lma_q2_glob_loop_nopre_f_048.cpp -o cpu_lma_q2_glob_loop_nopre_f_048.o
	$(CPP) $(CXX_FLAGS) -c ../cpu_lma_q2_glob_loop_nopre_d_049.cpp -o cpu_lma_q2_glob_loop_nopre_d_049.o
	$(CPP) $(CXX_FLAGS) -c  ../cpu_lma_q2_loc_loop_nopre_f_056.cpp -o  cpu_lma_q2_loc_loop_nopre_f_056.o
	$(CPP) $(CXX_FLAGS) -c  ../cpu_lma_q2_loc_loop_nopre_d_057.cpp -o  cpu_lma_q2_loc_loop_nopre_d_057.o

OBJ_CPP = vectorcpu.o
OBJ_CPP_RUN = cpu_csr_q1_glob_loop_nopre_f_000.o \
			  cpu_csr_q1_glob_loop_nopre_d_001.o \
			   cpu_csr_q1_loc_loop_nopre_f_008.o \
			   cpu_csr_q1_loc_loop_nopre_d_009.o \
			  cpu_csr_q2_glob_loop_nopre_f_016.o \
			  cpu_csr_q2_glob_loop_nopre_d_017.o \
			   cpu_csr_q2_loc_loop_nopre_f_024.o \
			  cpu_csr_q2_loc_loop_nopre_d_025.o  \
			  cpu_lma_q1_glob_loop_nopre_f_032.o \
			  cpu_lma_q1_glob_loop_nopre_d_033.o \
			   cpu_lma_q1_loc_loop_nopre_f_040.o \
			   cpu_lma_q1_loc_loop_nopre_d_041.o \
			  cpu_lma_q2_glob_loop_nopre_f_048.o \
			  cpu_lma_q2_glob_loop_nopre_d_049.o \
			  cpu_lma_q2_loc_loop_nopre_f_056.o  \
			  cpu_lma_q2_loc_loop_nopre_d_057.o

cpu_run: cpu_compile
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) cpu_csr_q1_glob_loop_nopre_f_000.o -o cpu_csr_q1_glob_loop_nopre_f_000
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) cpu_csr_q1_glob_loop_nopre_d_001.o -o cpu_csr_q1_glob_loop_nopre_d_001
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP)  cpu_csr_q1_loc_loop_nopre_f_008.o -o  cpu_csr_q1_loc_loop_nopre_f_008
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP)  cpu_csr_q1_loc_loop_nopre_d_009.o -o  cpu_csr_q1_loc_loop_nopre_d_009
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) cpu_csr_q2_glob_loop_nopre_f_016.o -o cpu_csr_q2_glob_loop_nopre_f_016
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) cpu_csr_q2_glob_loop_nopre_d_017.o -o cpu_csr_q2_glob_loop_nopre_d_017
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP)  cpu_csr_q2_loc_loop_nopre_f_024.o -o  cpu_csr_q2_loc_loop_nopre_f_024
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP)  cpu_csr_q2_loc_loop_nopre_d_025.o -o  cpu_csr_q2_loc_loop_nopre_d_025
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) cpu_lma_q1_glob_loop_nopre_f_032.o -o cpu_lma_q1_glob_loop_nopre_f_032
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) cpu_lma_q1_glob_loop_nopre_d_033.o -o cpu_lma_q1_glob_loop_nopre_d_033
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP)  cpu_lma_q1_loc_loop_nopre_f_040.o -o  cpu_lma_q1_loc_loop_nopre_f_040
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP)  cpu_lma_q1_loc_loop_nopre_d_041.o -o  cpu_lma_q1_loc_loop_nopre_d_041
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) cpu_lma_q2_glob_loop_nopre_f_048.o -o cpu_lma_q2_glob_loop_nopre_f_048
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP) cpu_lma_q2_glob_loop_nopre_d_049.o -o cpu_lma_q2_glob_loop_nopre_d_049
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP)  cpu_lma_q2_loc_loop_nopre_f_056.o -o  cpu_lma_q2_loc_loop_nopre_f_056
	$(CPP) $(CXX_FLAGS) $(OBJ_CPP)  cpu_lma_q2_loc_loop_nopre_d_057.o -o  cpu_lma_q2_loc_loop_nopre_d_057

RUN_FILES_CPU = cpu_csr_q1_glob_loop_nopre_f_000 \
				cpu_csr_q1_glob_loop_nopre_d_001 \
				 cpu_csr_q1_loc_loop_nopre_f_008 \
				 cpu_csr_q1_loc_loop_nopre_d_009 \
				cpu_csr_q2_glob_loop_nopre_f_016 \
				cpu_csr_q2_glob_loop_nopre_d_017 \
				 cpu_csr_q2_loc_loop_nopre_f_024 \
				 cpu_csr_q2_loc_loop_nopre_d_025 \
				cpu_lma_q1_glob_loop_nopre_f_032 \
				cpu_lma_q1_glob_loop_nopre_d_033 \
				 cpu_lma_q1_loc_loop_nopre_f_040 \
				 cpu_lma_q1_loc_loop_nopre_d_041 \
				cpu_lma_q2_glob_loop_nopre_f_048 \
				cpu_lma_q2_glob_loop_nopre_d_049 \
				 cpu_lma_q2_loc_loop_nopre_f_056 \
				 cpu_lma_q2_loc_loop_nopre_d_057

gpu_compile: ../../../csrmatrixgpu.cpp \
			 ../../../csrmatrixgpu.cu  \
			 ../../../basics.cu        \
			  ../gpu_csr_q1_loc_loop_nopre_f_072.cpp \
								   ../kernel_072.cu
	$(CUDA) $(CUDA_FLAGS) -c ../../../csrmatrixgpu.cpp -o csrmatrixgpu.o
	$(CUDA) $(CUDA_FLAGS) -c ../../../csrmatrixgpu.cu  -o csrmatrixgpu_c.o
	$(CUDA) $(CUDA_FLAGS) -c ../../../basics.cu        -o basics_c.o
	$(CPP) $(CXX_FLAGS) $(CUDA_LIBS) -c  ../gpu_csr_q1_loc_loop_nopre_f_072.cpp -o gpu_csr_q1_loc_loop_nopre_f_072.o
	$(CUDA) $(CUDA_FLAGS)            -c                       ../kernel_072.cu  -o                      kernel_072_c.o

OBJ_CUDA = csrmatrixgpu.o   \
		   csrmatrixgpu_c.o \
		   basics_c.o
OBJ_CUDA_RUN =  gpu_csr_q1_loc_loop_nopre_f_072.o   \
			                         kernel_072_c.o

gpu_run: cpu_compile gpu_compile
	$(CPP) $(CXX_FLAGS) $(CUDA_LIBS) $(OBJ_CUDA) $(OBJ_CPP)  gpu_csr_q1_loc_loop_nopre_f_072.o kernel_072_c.o -o  gpu_csr_q1_loc_loop_nopre_f_072

RUN_FILES_CUDA = gpu_csr_q1_loc_loop_nopre_f_072

all: cpu_run gpu_run

clean:
	rm -rf $(OBJ_CPP)  $(OBJ_CPP_RUN)
	rm -rf $(OBJ_CUDA) $(OBJ_CUDA_RUN)

cleanfull: clean
	rm -rf $(RUN_FILES_CPP)
	rm -rf $(RUN_FILES_CUDA)
