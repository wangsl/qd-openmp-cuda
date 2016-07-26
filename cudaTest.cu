
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

__global__ void _my_set_data_(const int n, Complex *c, const double v)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n)
    c[index] = Complex(v, 0.0);
}

__global__ void _check_data_(const int n, Complex *c1, const Complex *c2)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n)
    c1[index] = c2[index];
}

void omegas_test(OmegaStates &omegas)
{
  std::cout << " Omega Test" << std::endl;
  
  std::cout << omegas.wave_packets.size() << std::endl;

#if 1
  cudaPointerAttributes wp_attr;
  if(cudaPointerGetAttributes(&wp_attr, &omegas.wave_packets[0][0]) == cudaErrorInvalidValue)  {
    const int &n1 = omegas.wave_packets.size();
    const int &n2 = omegas.wave_packets[0].size();
    std::cout << " Paged memory" << std::endl;
    checkCudaErrors(cudaHostRegister(&omegas.wave_packets[0][0], n1*n2*sizeof(Complex), 
				     cudaHostRegisterPortable));
  }
  
  if(cudaPointerGetAttributes(&wp_attr, &omegas.wave_packets[0][0]) == cudaErrorInvalidValue)  {
    std::cout << " Paged memory" << std::endl;
  } else {
    std::cout << " UnPaged memory" << std::endl;
  }
#endif
  
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

  for(int k = 0; k < 10; k++) {
    
    std::cout << " k = " << k << " " << time_now() << std::endl;
    
#pragma omp parallel default(shared)
    {
      const int i_th = omp_get_thread_num();
      
      checkCudaErrors(cudaSetDevice(i_th));
      
      int mine_dev = -1;
      checkCudaErrors(cudaGetDevice(&mine_dev));
      
      cudaStream_t *streams = (cudaStream_t *) malloc(n_gpus*sizeof(cudaStream_t));
      insist(streams);
      for(int i_dev = 0; i_dev < n_gpus; i_dev++) 
	checkCudaErrors(cudaStreamCreate(&(streams[i_dev])));
      
      const int n_threads = 512;
      const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
      
      for(int i_dev = mine_dev+1; i_dev < n_gpus; i_dev++) {
	
	checkCudaErrors(cudaStreamCreate(&(streams[i_dev])));
	
	if(i_dev != mine_dev) 
	  checkCudaErrors(cudaDeviceEnablePeerAccess(i_dev, 0));
	
	_check_data_<<<n_blocks, n_threads, 0, streams[i_dev]>>>(n, x_dev[i_dev].data2, x_dev[mine_dev].data);
	_check_data_<<<n_blocks, n_threads, 0, streams[i_dev]>>>(n, x_dev[mine_dev].data2, x_dev[i_dev].data);
	
	if(i_dev != mine_dev) 
	  checkCudaErrors(cudaDeviceDisablePeerAccess(i_dev));
      }
      
      for(int i_dev = 0; i_dev < n_gpus; i_dev++) 
	checkCudaErrors(cudaStreamDestroy(streams[i_dev]));
      
      if(streams) { free(streams); streams = 0; }
    }
  }
  
  std::cout << time_now() << std::endl;
  
  for(int i = 0; i < n_gpus; i++) {
    std::cout << i << " " << x_dev[i].gpu_index << " " << x_dev[i].data << " " << x_dev[i].data2 << std::endl;
    if(x_dev[i].data) checkCudaErrors(cudaFree(x_dev[i].data));
    if(x_dev[i].data2) checkCudaErrors(cudaFree(x_dev[i].data2));
  }

  //std::this_thread::sleep_for(std::chrono::milliseconds(60*1000));
  
  if(x_dev) { delete [] x_dev; x_dev = 0; }
}

#if 0
void my_p2p_test()
{
  const int n = 10240;
  
  const int n_threads = 256;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);

  std::cout << " n_blocks: " << n_blocks << std::endl;
  
  double *c_dev = 0;
  checkCudaErrors(cudaMalloc(&c_dev, n*sizeof(double)));
  insist(c_dev);
  checkCudaErrors(cudaMemset(c_dev, 0, n*sizeof(double)));
  
  const double x = 2.0;
  double *x_dev = 0;
  checkCudaErrors(cudaMalloc(&x_dev, sizeof(double)));
  insist(x_dev);
  checkCudaErrors(cudaMemcpy(x_dev, &x, sizeof(double), cudaMemcpyHostToDevice));

  _my_set_data_<<<n_blocks, n_threads>>>(c_dev, x_dev);
      
  double c[n];
  memset(c, 0, n*sizeof(double));
  
  for(int i = 0; i < 10; i++)
    std::cout << c[i] << std::endl;
  
  checkCudaErrors(cudaMemcpy(c, c_dev, n*sizeof(double), cudaMemcpyDeviceToHost));

  for(int i = 0; i < 10; i++)
    std::cout << c[i] << std::endl;

}
#endif

static __global__ void _print_data_(const int n, const Complex *c1, const Complex *c2)
{
  for(int i = 0; i < 10; i++)
    printf("%.4f %.4f %.4f %.4f\n", c1[i].re, c1[i].im, c2[i].re, c2[i].im);
}

void my_p2p_test_2()
{
  int n_gpus = 0;
  checkCudaErrors(cudaGetDeviceCount(&n_gpus));
  
  std::cout << " There are " << n_gpus << " GPU cards" << std::endl;
  
  DevArray *x_dev = new DevArray [n_gpus];
  insist(x_dev);
  
  omp_set_num_threads(n_gpus);

  const int n = 102400;
  
  const int n_threads = 512;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
  
  std::cout << time_now() << std::endl;

#pragma omp parallel default(shared)
  {
    const int i_th = omp_get_thread_num();
    
    checkCudaErrors(cudaSetDevice(i_th));
    
    int i_dev = -1;
    checkCudaErrors(cudaGetDevice(&i_dev));
    
    x_dev[i_th].gpu_index = i_dev;
    
    if(!x_dev[i_th].data) { 
      checkCudaErrors(cudaMalloc(&(x_dev[i_th].data), n*sizeof(Complex)));
      insist(x_dev[i_th].data);
    }
    
    _my_set_data_<<<n_blocks, n_threads>>>(n, x_dev[i_th].data, i_dev);
  }
  
  std::cout << time_now();
  
  checkCudaErrors(cudaSetDevice(1));
  checkCudaErrors(cudaDeviceEnablePeerAccess(0, 0));
  
  _print_data_<<<1,1>>>(n, x_dev[0].data, x_dev[1].data);

#pragma omp parallel for default(shared)
  for(int i = 0; i < n_gpus; i++) {
    //std::cout << i << " " << x_dev[i].gpu_index << " " << x_dev[i].data << std::endl;
    checkCudaErrors(cudaSetDevice(i));
    //_print_data_<<<1,1>>>(n, x_dev[i].data, x_dev[i].data);
    if(x_dev[i].data) checkCudaErrors(cudaFree(x_dev[i].data));
  }

  if(x_dev) { delete [] x_dev; x_dev = 0; }

}


void omegas_test_2(OmegaStates &omegas)
{
  std::cout << " Omega Test 2" << std::endl;
  
  std::cout << omegas.wave_packets.size() << std::endl;

#if 0
  cudaPointerAttributes wp_attr;
  if(cudaPointerGetAttributes(&wp_attr, &omegas.wave_packets[0][0]) == cudaErrorInvalidValue)  {
    const int &n1 = omegas.wave_packets.size();
    const int &n2 = omegas.wave_packets[0].size();
    std::cout << " Paged memory" << std::endl;
    checkCudaErrors(cudaHostRegister(&omegas.wave_packets[0][0], n1*n2*sizeof(Complex), 
				     cudaHostRegisterPortable));
  }
  
  if(cudaPointerGetAttributes(&wp_attr, &omegas.wave_packets[0][0]) == cudaErrorInvalidValue)  {
    std::cout << " Paged memory" << std::endl;
  } else {
    std::cout << " UnPaged memory" << std::endl;
  }
#endif
  
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

  insist(n_gpus == 8);

  const int loop_index [] = { 10, 32, 54, 76,
			      20, 31, 64, 75, 
			      30, 21, 65, 74,
			      40, 51, 62, 73,
			      50, 41, 63, 72,
			      60, 42, 53, 71,
			      70, 61, 52, 43};

  const int n_loops = sizeof(loop_index)/sizeof(int);

  std::cout << " n_loops = " << n_loops << std::endl;
  
  for(int k = 0; k < 10; k++) {
    
    std::cout << " k = " << k << " " << time_now() << std::endl;

    omp_set_num_threads(n_gpus/2);
    
#pragma omp parallel for default(shared) schedule(static,1) 
    
    for(int l = 0; l < n_loops; l++) {
      
      const int index = loop_index[l];
      const int dev_1 = index/10;
      const int dev_2 = index - 10*dev_1;

      if(omp_get_thread_num() == 0) 
	std::cout << " index " << index << std::endl;
      
      checkCudaErrors(cudaSetDevice(dev_1));
      
      const int n_threads = 512;
      const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
      
      checkCudaErrors(cudaDeviceEnablePeerAccess(dev_2, 0));
      
      _check_data_<<<n_blocks, n_threads>>>(n, x_dev[dev_1].data2, x_dev[dev_2].data);
      _check_data_<<<n_blocks, n_threads>>>(n, x_dev[dev_2].data2, x_dev[dev_1].data);
      
      checkCudaErrors(cudaDeviceDisablePeerAccess(dev_2));
    }
    
    //for(int i_dev = 0; i_dev < n_gpus; i_dev++) 
    //checkCudaErrors(cudaStreamDestroy(streams[i_dev]));
    
    //if(streams) { free(streams); streams = 0; }
  }
  
  std::cout << time_now();
  
  for(int i = 0; i < n_gpus; i++) {
    std::cout << i << " " << x_dev[i].gpu_index << " " << x_dev[i].data << " " << x_dev[i].data2 << std::endl;
    if(x_dev[i].data) checkCudaErrors(cudaFree(x_dev[i].data));
    if(x_dev[i].data2) checkCudaErrors(cudaFree(x_dev[i].data2));
  }

  //std::this_thread::sleep_for(std::chrono::milliseconds(60*1000));
  
  if(x_dev) { delete [] x_dev; x_dev = 0; }
}

void omegas_test_3(OmegaStates &omegas)
{
  std::cout << " Omega Test 3" << std::endl;
  
  std::cout << omegas.wave_packets.size() << std::endl;
  
#if 1
  cudaPointerAttributes wp_attr;
  if(cudaPointerGetAttributes(&wp_attr, &omegas.wave_packets[0][0]) == cudaErrorInvalidValue)  {
    const int &n1 = omegas.wave_packets.size();
    const int &n2 = omegas.wave_packets[0].size();
    std::cout << " Paged memory" << std::endl;
    checkCudaErrors(cudaHostRegister(&omegas.wave_packets[0][0], n1*n2*sizeof(Complex), 
				     cudaHostRegisterPortable));
  }
  
  if(cudaPointerGetAttributes(&wp_attr, &omegas.wave_packets[0][0]) == cudaErrorInvalidValue)  {
    std::cout << " Paged memory" << std::endl;
  } else {
    std::cout << " UnPaged memory" << std::endl;
  }
#endif
  
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
      
      checkCudaErrors(cudaSetDevice(i_dev));
      
      cudaStream_t *streams = (cudaStream_t *) malloc(n_gpus*sizeof(cudaStream_t));
      insist(streams);

      for(int j_dev = 0; j_dev < n_gpus; j_dev++) {
	
	if(j_dev != i_dev)  {
	  
	  checkCudaErrors(cudaDeviceEnablePeerAccess(j_dev, 0));
	  
	  checkCudaErrors(cudaStreamCreate(&(streams[j_dev])));
	  
	  checkCudaErrors(cudaMemcpyPeerAsync(x_dev[j_dev].data2, j_dev, 
					      x_dev[i_dev].data, i_dev,
					      n*sizeof(Complex), streams[j_dev]));
	}
      }
      
      for(int j_dev = 0; j_dev < n_gpus; j_dev++) {
	if(j_dev != i_dev) {
	  checkCudaErrors(cudaStreamSynchronize(streams[j_dev]));
	  checkCudaErrors(cudaStreamDestroy(streams[j_dev]));
	  checkCudaErrors(cudaDeviceDisablePeerAccess(j_dev));
	}
      }
      
      if(streams) { free(streams); streams = 0; }
    }
  }
  
  std::cout << time_now();
  
  for(int i = 0; i < n_gpus; i++) {
    std::cout << i << " " << x_dev[i].gpu_index << " " << x_dev[i].data << " " << x_dev[i].data2 << std::endl;
    if(x_dev[i].data) checkCudaErrors(cudaFree(x_dev[i].data));
    if(x_dev[i].data2) checkCudaErrors(cudaFree(x_dev[i].data2));
  }

  if(x_dev) { delete [] x_dev; x_dev = 0; }
}

void omegas_test_4(OmegaStates &omegas)
{
  std::cout << " Omega Test 4" << std::endl;
  
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
	
	checkCudaErrors(cudaMemcpyPeerAsync(x_dev[j_dev].data2, j_dev, 
					    x_dev[i_dev].data, i_dev,
					    n*sizeof(Complex), streams[i_stream]));
	i_stream++;
	
	checkCudaErrors(cudaSetDevice(j_dev));
	checkCudaErrors(cudaDeviceEnablePeerAccess(i_dev, 0));
	checkCudaErrors(cudaStreamCreate(&(streams[i_stream])));
	
	checkCudaErrors(cudaMemcpyPeerAsync(x_dev[i_dev].data2, i_dev, 
					    x_dev[j_dev].data, j_dev,
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

void omegas_test_5(OmegaStates &omegas)
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
