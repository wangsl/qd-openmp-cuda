
#include <iostream>
#include <cstring>
#include <omp.h>
#include <cassert>
#include <helper_cuda.h>

#include <chrono>
#include <thread>

#include "cudaUtils.h"
#include "cudaMath.h"
#include "matlabStructures.h"

#ifdef printf
#undef printf
#endif

#define _NLOOPS_ 10

struct DevArray
{
  Complex *data = 0;
  int gpu_index = -1;
  Complex *data2 = 0;
};

void omegas_test(OmegaStates &omegas)
{
  std::cout << " Omega Test 5" << std::endl;
  
  std::cout << omegas.wave_packets.size() << std::endl;
  
  int n_gpus = 0;
  checkCudaErrors(cudaGetDeviceCount(&n_gpus));
  
  std::cout << " There are " << n_gpus << " GPU cards" << std::endl;
  
  DevArray *x_dev = new DevArray [n_gpus];
  insist(x_dev);
  
  omp_set_num_threads(n_gpus);

  const int n = omegas.wave_packets[0].size();

  std::cout << time_now() << std::endl;
  
#pragma omp parallel default(shared)
  {
    const int i_th = omp_get_thread_num();
    
    const int n = omegas.wave_packets[i_th].size();
    if(i_th == 0) std::cout << " n = " << n << std::endl;
    
    checkCudaErrors(cudaSetDevice(i_th));
    
    int i_dev = -1;
    checkCudaErrors(cudaGetDevice(&i_dev));
    
    x_dev[i_th].gpu_index = i_dev;
      
    if(!x_dev[i_th].data) { 
      checkCudaErrors(cudaMalloc(&(x_dev[i_th].data), n*sizeof(Complex)));
      insist(x_dev[i_th].data);
    }
    
    if(!x_dev[i_th].data2) {
      checkCudaErrors(cudaMalloc(&(x_dev[i_th].data2), n*sizeof(Complex)));
      insist(x_dev[i_th].data2);
    }
    
    checkCudaErrors(cudaMemcpy(x_dev[i_th].data, (const Complex *) omegas.wave_packets[i_th],
			       n*sizeof(Complex), cudaMemcpyHostToDevice));
  }

  std::cout << time_now() << std::endl;
  
  for(int k = 0; k < _NLOOPS_; k++) {
    
    std::cout << " k = " << k << " " << time_now() << std::endl;
    
#pragma omp parallel for default(shared)
    for(int i_dev = 0; i_dev < n_gpus; i_dev++) {
      
      cudaStream_t *streams = (cudaStream_t *) malloc(2*(n_gpus-i_dev-1)*sizeof(cudaStream_t));
      insist(streams);
      
      int i_stream = 0;
      for(int j_dev = i_dev+1; j_dev < n_gpus; j_dev++) {

	checkCudaErrors(cudaSetDevice(i_dev));
	checkCudaErrors(cudaDeviceEnablePeerAccess(j_dev, 0));
	checkCudaErrors(cudaStreamCreate(&(streams[i_stream])));
	checkCudaErrors(cudaMemcpyPeerAsync(x_dev[i_dev].data2, i_dev, 
					    x_dev[j_dev].data, j_dev,
					    n*sizeof(Complex), streams[i_stream]));
	i_stream++;
	
	checkCudaErrors(cudaSetDevice(j_dev));
	checkCudaErrors(cudaDeviceEnablePeerAccess(i_dev, 0));
	checkCudaErrors(cudaStreamCreate(&(streams[i_stream])));
	checkCudaErrors(cudaMemcpyPeerAsync(x_dev[j_dev].data2, j_dev, 
					    x_dev[i_dev].data, i_dev,
					    n*sizeof(Complex), streams[i_stream]));
	i_stream++;
      }
      
      insist(i_stream == 2*(n_gpus-i_dev-1));
      
      i_stream = 0;
      for(int j_dev = i_dev+1; j_dev < n_gpus; j_dev++) {
	checkCudaErrors(cudaSetDevice(i_dev));
	checkCudaErrors(cudaStreamSynchronize(streams[i_stream]));
	checkCudaErrors(cudaStreamDestroy(streams[i_stream]));
	i_stream++;
	checkCudaErrors(cudaDeviceDisablePeerAccess(j_dev));
	
	checkCudaErrors(cudaSetDevice(j_dev));
	checkCudaErrors(cudaStreamSynchronize(streams[i_stream]));
	checkCudaErrors(cudaStreamDestroy(streams[i_stream]));
	i_stream++;
	checkCudaErrors(cudaDeviceDisablePeerAccess(i_dev));
      }
      
      if(streams) { free(streams); streams = 0; }
    }
  }
  
  std::cout << time_now() << std::endl;
  
  for(int i = 0; i < n_gpus; i++) {
    std::cout << i << " " << x_dev[i].gpu_index 
	      << " " << x_dev[i].data << " " << x_dev[i].data2 << std::endl;
    if(x_dev[i].data) checkCudaErrors(cudaFree(x_dev[i].data));
    if(x_dev[i].data2) checkCudaErrors(cudaFree(x_dev[i].data2));
  }

  if(x_dev) { delete [] x_dev; x_dev = 0; }
}


