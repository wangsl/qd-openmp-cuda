
#ifndef CUDA_OPENMP_H
#define CUDA_OPENMP_H

#include <iostream>
#include <cstring>
#include <omp.h>
#include <cassert>

#include "matlabStructures.h"
#include "wavepackets1device.h"
#include "evolutionUtils.h"

class CudaOpenMPQMMD
{
public:
  CudaOpenMPQMMD(); 
  
  ~CudaOpenMPQMMD();
  
  const int &n_gpus() const { return _n_gpus; }

  void test();

  void test_coriolis() const;

  Vec<OmegaWavepacketsOnSingleDevice *> omega_wavepackets_on_single_device;

  Vec<CoriolisMatrixAux> coriolis_matrices;

private:
  
  int _n_gpus;
  
  void setup_n_gpus();
  
  void setup_wavepackets_on_single_device();
  void destroy_wavepackets_on_single_device();
  
  void devices_synchronize();
  void devices_memory_usage() const;
  
  void setup_coriolis_matrices_and_copy_to_device();
  
  void setup_constant_memory_on_device();

  void reset_devices();

  void copy_psi_from_device_to_host();
};

#endif /* CUDA_OPENMP_H */
