
#include "cudaOpenMP.h"

CudaOpenMPQMMD::CudaOpenMPQMMD(const double *pot_,
			       const RadialCoordinate &r1_,
			       const RadialCoordinate &r2_,
			       const AngleCoordinate &theta_,
			       OmegaStates &omegas_,
			       EvolutionTime &time_) :
  _n_gpus(0),
  pot(pot_), 
  r1(r1_), r2(r2_), theta(theta_), omegas(omegas_), time(time_)
{
  setup_n_gpus();
  setup_wavepackets_on_single_device();
}

CudaOpenMPQMMD::~CudaOpenMPQMMD()
{
  pot = 0;
  destroy_wavepackets_on_single_device();
}
