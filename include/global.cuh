#ifndef __GLOBAL_CUH_
#define __GLOBAL_CUH_

void get_kernel_config(dim3* const numblocks, dim3* const numthreads, size_t totalthreads);

#endif
