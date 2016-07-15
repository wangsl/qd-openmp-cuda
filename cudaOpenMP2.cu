
#include <chrono>
#include <ctime>
#include <helper_cuda.h>

#include "cudaUtils.h"
#include "cudaOpenMP.h"

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
  for(int L = 0; L < time.total_steps; L++) {
 
    std::cout << std::endl << " L: " << L << ", " << time_now() << std::endl;
    
    omp_set_num_threads(n_gpus());
    
#pragma omp parallel for default(shared)
    for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
      omega_wavepackets_on_single_device[i_dev]->calculate_omega_wavepacket_modules();
      omega_wavepackets_on_single_device[i_dev]->calculate_omega_wavepacket_potential_energy();
      omega_wavepackets_on_single_device[i_dev]->forward_legendre_transform();
      
      omega_wavepackets_on_single_device[i_dev]->forward_fft_for_legendre_psi();
      omega_wavepackets_on_single_device[i_dev]->backward_fft_for_legendre_psi(1);

      omega_wavepackets_on_single_device[i_dev]->backward_legendre_transform();
    }

    std::ios_base::fmtflags old_flags = std::cout.flags();
    
    for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
      const int &n = omega_wavepackets_on_single_device[i_dev]->omega_wavepackets.size();
      for(int i = 0; i < n; i++)
	std::cout << " " << i_dev << " " << i 
		  << std::fixed
		  << " " << omega_wavepackets_on_single_device[i_dev]->omega_wavepackets[i]->wavepacket_module() 
		  << " " << omega_wavepackets_on_single_device[i_dev]->omega_wavepackets[i]->potential_energy()
		  << std::endl;
    }

    std::cout.flags(old_flags);
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
  
  omegas_index.show_in_one_line();

  int omega_start_index = 0;
  for(int i_dev = 0; i_dev < n; i_dev++) {
    const int n_omegas = omegas_index[i_dev];
    omega_wavepackets_on_single_device[i_dev] = \
      new OmegaWavepacketsOnSingleDevice(i_dev, omega_start_index, n_omegas, 
					 pot, r1, r2, theta, omegas, omegas.l_max);
    
    insist(omega_wavepackets_on_single_device[i_dev]);
    
    omega_start_index += n_omegas;
  }

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
