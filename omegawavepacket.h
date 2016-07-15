
#ifndef OMEGA_WAVEPACKET_H
#define OMEGA_WAVEPACKET_H

#include <cublas_v2.h>
#include "matlabStructures.h"
#include "complex.h"

class OmegaWavepacket
{

public:

  OmegaWavepacket(const int &omega,
		  const int &l_max,
		  const RMat &associated_legendres,
		  const RadialCoordinate &r1,
		  const RadialCoordinate &r2,
		  const AngleCoordinate &theta,
		  Complex *psi, 
		  const double *pot_dev,
		  cublasHandle_t &cublas_handle,
		  cufftHandle &cufft_plan_for_legendre_psi,
		  Complex * &work_dev
		  );


  ~OmegaWavepacket();

  const double &wavepacket_module() const  { return _wavepacket_module; }
  const double &potential_energy() const { return _potential_energy; }

  void calculate_wavepacket_module();
  void calculate_potential_energy();
  
  void forward_legendre_transform();
  void backward_legendre_transform();
  
  void forward_fft_for_legendre_psi();
  void backward_fft_for_legendre_psi(const int do_scale = 0);
  
private:
  
  Complex *psi;

  const int &omega;
  const int &l_max;
  const RMat &associated_legendres;

  const RadialCoordinate &r1;
  const RadialCoordinate &r2;
  const AngleCoordinate &theta;
  
  const double *pot_dev;
  Complex *psi_dev;
  Complex *legendre_psi_dev;
  Complex *associated_legendres_dev;
  Complex *weighted_associated_legendres_dev;

  Complex * &work_dev;

  cublasHandle_t &cublas_handle;
  cufftHandle &cufft_plan_for_legendre_psi;

  double _wavepacket_module;
  double _potential_energy;

  void setup_device_data();
  void setup_associated_legendres();
  void setup_weighted_associated_legendres();
  void setup_legendre_psi();
};

#endif /* OMEG_WAVEPACKET_H */
