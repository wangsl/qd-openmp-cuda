
#ifndef OMEGA_WAVEPACKET_H
#define OMEGA_WAVEPACKET_H

#include <cublas_v2.h>
#include "matlabStructures.h"
#include "complex.h"
#include "evolutionUtils.h"

class OmegaWavepacket
{

public:

  friend class OmegaWavepacketsOnSingleDevice;

  OmegaWavepacket(const int &omega,
		  const int &l_max,
		  const Vec<CoriolisMatrixAux> &coriolis_matrices,
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
  const double &kinetic_energy() const { return _kinetic_energy; }
  const double &rotational_energy() const { return _rotational_energy; }
  const double &coriolis_energy() const { return _coriolis_energy; }
  
  double total_energy() const
  { return potential_energy() + kinetic_energy() + rotational_energy() + coriolis_energy(); }
  
  const double &wavepacket_module_for_legendre_psi() const      
  { return _wavepacket_module_for_legendre_psi; }

private:

  void calculate_wavepacket_module();
  void calculate_potential_energy();
  void calculate_kinetic_energy_for_legendre_psi();
  void calculate_rotational_energy_for_legendre_psi();
  
  void calculate_wavepacket_module_for_legendre_psi();
  
  void calculate_coriolis_energy_for_legendre_psi(const int omega1,
						  const double *coriolis_matrices_dev,
						  const Complex *legendre_psi_omega1);
  
  void calculate_coriolis_energy_for_legendre_psi_2(const int omega1,
						    const double *coriolis_matrices_dev,
						    const Complex *legendre_psi_omega1);
  
  
  void forward_legendre_transform();
  void backward_legendre_transform();
  
  void forward_fft_for_legendre_psi();
  void backward_fft_for_legendre_psi(const int do_scale = 0);

  void copy_psi_from_host_to_device();
  void copy_psi_from_device_to_host();

private:
  
  Complex *psi;

  const int &omega;
  const int &l_max;
  const Vec<CoriolisMatrixAux> &coriolis_matrices;
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
  double _kinetic_energy;
  double _rotational_energy;
  double _coriolis_energy;

  double _wavepacket_module_for_legendre_psi;

  void setup_device_data();
  void setup_associated_legendres();
  void setup_weighted_associated_legendres();
  void setup_legendre_psi();

  void evolution_with_potential(const double dt);
  void evolution_with_kinetic(const double dt);
  void evolution_with_rotational(const double dt);
  
  void evolution_with_coriolis(const double dt,
			       const int l, const int omega1,
			       const double *coriolis_matrices_dev,
			       const Complex *legendre_psi_omega1,
			       cudaStream_t *stream = 0); 
  
  void evolution_with_coriolis(const double dt, const int omega1,
			       const double *coriolis_matrices_dev,
			       const Complex *legendre_psi_omega1);

  void zero_psi_dev();
  void update_evolution_with_coriolis();
};

#endif /* OMEG_WAVEPACKET_H */
