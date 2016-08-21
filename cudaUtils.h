
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <chrono>
#include <thread>

#define _CUDA_FREE_(x) if(x) { checkCudaErrors(cudaFree(x)); x = 0; }

#define _NTHREADS_ 512

namespace cudaUtils {
  
  inline int number_of_blocks(const int n_threads, const int n)
  { return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }
  
  void gpu_memory_usage();
}

inline char *time_now()
{
  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  std::time_t time_now  = std::chrono::system_clock::to_time_t(now);
  char *time = std::ctime(&time_now); 
  char *pch = strchr(time, '\n');
  if(pch) pch[0] = '\0';
  return time;
}

#endif /* CUDA_UTILS_H */
