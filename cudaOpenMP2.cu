
#include <chrono>
#include <ctime>
#include <helper_cuda.h>

#include "cudaUtils.h"
#include "cudaOpenMP.h"
#include "evolutionUtils.h"

inline void divide_into_chunks(const int n, const int m, int *chunks)
{
  memset(chunks, 0, m*sizeof(int));
  
  for(int i = 0; i < m; i++)
    chunks[i] = n/m;
  
  for(int i = 0; i < n-n/m*m; i++)
    chunks[i]++;
}

void CudaOpenMPQMMD::setup_n_gpus()
{
  if(!_n_gpus) {
    checkCudaErrors(cudaGetDeviceCount(&_n_gpus));
    std::cout << " There are " << n_gpus() << " cards" << std::endl;
  }
}

void CudaOpenMPQMMD::test()
{
  insist(n_gpus() == 1);
  
  std::cout << time << std::endl;
  
  //omega_wavepackets_on_single_device[0]->test_coriolis_matrices();

  //return;

  for(int L = 0; L < time.total_steps; L++) {
    
    std::cout << std::endl << " L: " << L << ", " << time_now() << std::endl;
    
    omega_wavepackets_on_single_device[0]->evolution_test(L, time.time_step);
  }
}

void CudaOpenMPQMMD::setup_wavepackets_on_single_device()
{
  insist(n_gpus() > 0);
  
  insist(omega_wavepackets_on_single_device.size() == 0);
  omega_wavepackets_on_single_device.resize(n_gpus(), 0);
  
  const int &n = omega_wavepackets_on_single_device.size();

  Vec<int> omegas_index(n_gpus());
  divide_into_chunks(omegas.omegas.size(), n, omegas_index);
  
  std::cout << " Omegas index: ";
  omegas_index.show_in_one_line();
  
  int omega_start_index = 0;
  for(int i_dev = 0; i_dev < n; i_dev++) {
    const int n_omegas = omegas_index[i_dev];
    omega_wavepackets_on_single_device[i_dev] = 
      new OmegaWavepacketsOnSingleDevice(i_dev, omega_start_index, n_omegas, 
					 pot, r1, r2, theta, omegas, omegas.l_max,
					 coriolis_matrices);
    
    insist(omega_wavepackets_on_single_device[i_dev]);
    
    omega_start_index += n_omegas;
  }

  setup_coriolis_matrices_and_copy_to_device();

  setup_constant_memory_on_device();
  
  devices_synchronize();
  devices_memory_usage();
}

void CudaOpenMPQMMD::destroy_wavepackets_on_single_device()
{
  const int &n = omega_wavepackets_on_single_device.size();
  for(int i = 0; i < n; i++) {
    if(omega_wavepackets_on_single_device[i]) {
      delete omega_wavepackets_on_single_device[i];
      omega_wavepackets_on_single_device[i] = 0; 
    }
  }
  omega_wavepackets_on_single_device.resize(0);
}

void CudaOpenMPQMMD::devices_synchronize()
{
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

void CudaOpenMPQMMD::devices_memory_usage() const
{
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    cudaUtils::gpu_memory_usage();
  }
}

void CudaOpenMPQMMD::setup_coriolis_matrices_and_copy_to_device()
{
  std::cout << " Setup Coriolis Matrices and copy to devices" << std::endl;
  
  const int m = omegas.l_max + 1;
  coriolis_matrices.resize(m);
  
  int s = 0;
  for(int i = omegas.omegas[0]; i < m; i++) {
    int &l = coriolis_matrices[i].l;
    int &omega_min = coriolis_matrices[i].omega_min;
    int &omega_max = coriolis_matrices[i].omega_max;
    coriolis_matrices[i].offset = s;
    l = i;
    calculate_coriolis_matrix_dimension(omegas.J, omegas.parity, l, omega_min, omega_max);
    const int n = omega_max - omega_min + 1;
    s += n*(n+1);
  }

  double *cor_mats = new double [s];
  insist(cor_mats);
  memset(cor_mats, 0, s*sizeof(double));
  
  for(int i = 0; i < m; i++) {
    const int &l = coriolis_matrices[i].l;
    if(l == -1) continue;
    
    const int &omega_min = coriolis_matrices[i].omega_min;
    const int &omega_max = coriolis_matrices[i].omega_max;
    const int n = omega_max - omega_min + 1;
    RMat cor_mat_(n, n+1, cor_mats+coriolis_matrices[i].offset);

    //std::cout << coriolis_matrices[i] << std::endl;
    
    setup_coriolis_matrix(omegas.J, omegas.parity, l, cor_mat_);
    //std::cout << cor_mat_ << std::endl;
  }
  
  omp_set_num_threads(n_gpus());
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    omega_wavepackets_on_single_device[i_dev]->copy_coriolis_matrices_to_device(cor_mats, s);
  }
  
  devices_synchronize();
  
  if(cor_mats) { delete [] cor_mats; cor_mats = 0; }
}

void CudaOpenMPQMMD::setup_constant_memory_on_device()
{
  omp_set_num_threads(n_gpus());
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    omega_wavepackets_on_single_device[i_dev]->setup_constant_memory_on_device(time.time_step);
  }
}

void CudaOpenMPQMMD::reset_devices()
{
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    checkCudaErrors(cudaDeviceReset());
  }
}

void CudaOpenMPQMMD::test_coriolis() const
{
  omega_wavepackets_on_single_device[0]->test_coriolis_matrices();
}
