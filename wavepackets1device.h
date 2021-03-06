
#ifndef WAVEPACKETS_ON_SINGLE_DEVICE
#define WAVEPACKETS_ON_SINGLE_DEVICE

#include <cublas_v2.h>
#include "omegawavepacket.h"
#include "evolutionUtils.h"

class OmegaWavepacketsOnSingleDevice
{
  friend class CudaOpenMPQMMD;

public:
  OmegaWavepacketsOnSingleDevice(const int device_index,
				 const int omega_start_index,
				 const int n_omegas,
				 const Vec<CoriolisMatrixAux> &coriolis_matrices);
  
  ~OmegaWavepacketsOnSingleDevice();

private:
  
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

  void setup_constant_memory_on_device(); 

  void evolution_test(const int step, const double dt);

  void test_coriolis_matrices() const;

  void dump_wavepacket();

  void print_energies(const int print=0);

  void zero_coriolis_variables();
  void zero_work_dev_2(cudaStream_t *stream = 0);

private:

  double total_module;
  double total_energy;

  int _device_index;
  int omega_start_index;
  int n_omegas;

  const Vec<CoriolisMatrixAux> &coriolis_matrices;
  
  double *pot_dev;
  Complex *work_dev;
  double *coriolis_matrices_dev;
  Complex *work_dev_2;

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

  void evolution_with_coriolis_on_same_device(const double dt);
  
  void evolution_with_coriolis(const double dt, const int omega1,
			       const Complex *legendre_psi_omega1,
			       cudaStream_t *stream = 0);
  
  void calculate_coriolis_energy_for_legendre_psi(const int omega1,
						  const Complex *legendre_psi_omega1,
						  cudaStream_t *stream = 0);

  void update_evolution_with_coriolis();

  void setup_work_dev_2();
};

#endif /* WAVEPACKETS_ON_SINGLE_DEVICE */

