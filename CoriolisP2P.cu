
#include <helper_cuda.h>
#include "cudaOpenMP.h"
#include "matlabData.h"

void CudaOpenMPQMMD::evolution_with_coriolis_with_p2p(const double dt, const int calculate_energy)
{
  if(n_gpus() == 1) return;
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_legs = MatlabData::omega_states()->l_max;
  const int &n_theta = MatlabData::theta()->n; 
  insist(n_theta >= n_legs);
  const size_t n = n1*n2*n_legs;
  
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    
    OmegaWavepacketsOnSingleDevice * &omega_wpts_i_dev = omega_wavepackets_on_single_device[i_dev];
    const Vec<OmegaWavepacket *> &omega_wpts_i = omega_wpts_i_dev->omega_wavepackets;
    
    for(int i = 0; i < omega_wpts_i.size(); i++) {
      
      const Complex *psi_i = omega_wpts_i[i]->legendre_psi_dev_pointer();

      for(int j_dev = 0; j_dev < n_gpus(); j_dev++) {
	
	if(i_dev == j_dev) continue;
	
	OmegaWavepacketsOnSingleDevice * &omega_wpts_j_dev = omega_wavepackets_on_single_device[j_dev];
	
	omega_wpts_j_dev->zero_work_dev_2();
	checkCudaErrors(cudaMemcpyPeer(omega_wpts_j_dev->work_dev_2, j_dev,
				       psi_i, i_dev, n*sizeof(Complex)));
	
	const int &omega_i = omega_wpts_i[i]->omega_value();
	omega_wpts_j_dev->evolution_with_coriolis(dt, omega_i, omega_wpts_j_dev->work_dev_2);
	
	if(calculate_energy) 
	  omega_wpts_j_dev->calculate_coriolis_energy_for_legendre_psi(omega_i,
								       omega_wpts_j_dev->work_dev_2);
      }
    }
  }
}

void CudaOpenMPQMMD::evolution_with_coriolis_with_p2p_async(const double dt, 
							    const int calculate_energy)
{
  if(n_gpus() == 1) return;
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_legs = MatlabData::omega_states()->l_max;
  const int &n_theta = MatlabData::theta()->n; 
  insist(n_theta >= n_legs);
  const size_t n = n1*n2*n_legs;

  setup_streams_and_events(1, 0);
  
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    
    OmegaWavepacketsOnSingleDevice * &omega_wpts_i_dev = omega_wavepackets_on_single_device[i_dev];
    const Vec<OmegaWavepacket *> &omega_wpts_i = omega_wpts_i_dev->omega_wavepackets;
    
    for(int i = 0; i < omega_wpts_i.size(); i++) {
      
      const Complex *psi_i = omega_wpts_i[i]->legendre_psi_dev_pointer();
      
      for(int j_dev = 0; j_dev < n_gpus(); j_dev++) {
	
	if(i_dev == j_dev) continue;
	
	OmegaWavepacketsOnSingleDevice * &omega_wpts_j_dev = omega_wavepackets_on_single_device[j_dev];

	omega_wpts_j_dev->zero_work_dev_2();
	checkCudaErrors(cudaMemcpyPeer(omega_wpts_j_dev->work_dev_2, j_dev,
				       psi_i, i_dev, n*sizeof(Complex)));

	omega_wpts_j_dev->evolution_with_coriolis(dt, omega_wpts_i[i]->omega_value(),
						  omega_wpts_j_dev->work_dev_2, &streams[j_dev]);

	if(calculate_energy) {
	  omega_wpts_j_dev->calculate_coriolis_energy_for_legendre_psi(omega_wpts_i[i]->omega_value(),
								       omega_wpts_j_dev->work_dev_2, 
								       &streams[j_dev]);
	}
      }
    }
  }
}
