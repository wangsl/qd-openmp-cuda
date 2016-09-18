
#include "cudaOpenMP.h"

#if 0

CudaOpenMPQMMD::CudaOpenMPQMMD() :
  _n_gpus(0)
{
  setup_n_gpus();
  setup_wavepackets_on_single_device();
}

CudaOpenMPQMMD::~CudaOpenMPQMMD()
{
  std::cout << std::endl;
  destroy_wavepackets_on_single_device();
  reset_devices();
}

#endif
