
#ifndef WAVEPACKETS_ON_SINGLE_DEVICE
#define WAVEPACKETS_ON_SINGLE_DEVICE

#include <cublas_v2.h>
#include "omegawavepacket.h"
#include "evolutionUtils.h"

class OmegaWavepacketsOnSingleDevice
{
public:
  OmegaWavepacketsOnSingleDevice(const int device_index,
				 const int omega_start_index,
				 const int n_omegas,
				 const double * &pot,
				 const RadialCoordinate &r1,
				 const RadialCoordinate &r2,
				 const AngleCoordinate &theta,
				 OmegaStates &omegas_states,
				 const int &l_max,
				 const Vec<CoriolisMatrixAux> &coriolis_matrices);
  
  ~OmegaWavepacketsOnSingleDevice();
  
  Vec<OmegaWavepacket *> omega_wavepackets;
  
  const int &device_index() const { return _device_index; }
  int current_device_index() const;

  void calculate_wavepacket_modules();
  void calculate_wavepacket_potential_energies();

  void calculate_kinetic_energies_for_legendre_psi();
  void calculate_rotational_energies_for_legendre_psi();
  
  void calculate_wavepacket_modules_for_legendre_psi();

  void calculate_coriolis_energy_for_legendre_psi();

  void forward_legendre_transform();
  void backward_legendre_transform();

  void forward_fft_for_legendre_psi();
  void backward_fft_for_legendre_psi(const int do_scale);

  void copy_psi_from_device_to_host();

  void copy_coriolis_matrices_to_device(const double *c, const int s);

  void setup_constant_memory_on_device(); //const double time_step);

  void evolution_test(const int step, const double dt);

  void test_coriolis_matrices() const;

  void dump_wavepacket();

private:

  double total_module;
  double total_energy;

  int _device_index;
  int omega_start_index;
  int n_omegas;

  const double * &pot;

  const RadialCoordinate &r1;
  const RadialCoordinate &r2;
  const AngleCoordinate &theta;
  OmegaStates &omega_states;

  const int &l_max;
  const Vec<CoriolisMatrixAux> &coriolis_matrices;
  
  double *pot_dev;
  
  Complex *work_dev;

  double *coriolis_matrices_dev;

  cublasHandle_t cublas_handle;
  int has_cublas_handle;
  void setup_cublas_handle();
  void destroy_cublas_handle();

  cufftHandle cufft_plan_for_legendre_psi;
  int has_cufft_plan_for_legendre_psi;
  void setup_cufft_plan_for_legendre_psi();
  void destroy_cufft_plan_for_legendre_psi();

  void setup_device() const;
  
  void setup_constant_data_on_device();

  void setup_data_on_device();
  void destroy_data_on_device();

  void setup_potential_on_device();
  void setup_omega_wavepackets();

  void evolution_with_potential(const double dt);
  void evolution_with_kinetic(const double dt);
  void evolution_with_rotational(const double dt);
  void evolution_with_coriolis(const double dt);
};

#endif /* WAVEPACKETS_ON_SINGLE_DEVICE */

