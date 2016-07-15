
#ifndef CUDA_OPENMP_H
#define CUDA_OPENMP_H

#include <iostream>
#include <cstring>
#include <omp.h>
#include <cassert>

#include "matlabStructures.h"
#include "wavepackets1device.h"

class CudaOpenMPQMMD
{
public:
  CudaOpenMPQMMD(const double *pot, 
		 const RadialCoordinate &r1,
		 const RadialCoordinate &r2,
		 const AngleCoordinate &theta,
		 OmegaStates &omegas,
		 EvolutionTime &time);

  ~CudaOpenMPQMMD();

  const int &n_gpus() const { return _n_gpus; }

  void test();

  Vec<OmegaWavepacketsOnSingleDevice *> omega_wavepackets_on_single_device;
		 
private:
  
  int _n_gpus;
  const double *pot;
  
  const RadialCoordinate &r1;
  const RadialCoordinate &r2;
  const AngleCoordinate &theta;
  OmegaStates &omegas;
  EvolutionTime &time;

  void setup_n_gpus();

  void setup_wavepackets_on_single_device();
  void destroy_wavepackets_on_single_device();

  void devices_synchronize();
  void devices_memory_usage() const;
};

#endif /* CUDA_OPENMP_H */
