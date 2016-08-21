
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <helper_functions.h> 
#include <helper_cuda.h>
#include "cudaUtils.h"

void cudaUtils::gpu_memory_usage()
{
  std::ios_base::fmtflags old_flags = std::cout.flags();
  std::streamsize old_precision = std::cout.precision();
  
  std::cout.precision(2);
  
  int device_index = -1;
  checkCudaErrors(cudaGetDevice(&device_index));
  
  size_t free_byte = 0;
  size_t total_byte = 0;
  checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));
  
  std::cout << " Device: " << device_index
	    << " GPU memory usage:" 
	    << " used = " << std::fixed << (total_byte-free_byte)/1024.0/1024.0 << "MB,"
	    << " free = " << free_byte/1024.0/1024.0 << "MB,"
	    << " total = " << total_byte/1024.0/1024.0 << "MB" << std::endl;

  std::cout.flags(old_flags);
  std::cout.precision(old_precision);
}

