
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
  
  void test();
  
  void test_coriolis() const;
  
  void test_multiple_cards();

private:

  int _n_gpus;
  
  Vec<OmegaWavepacketsOnSingleDevice *> omega_wavepackets_on_single_device;
  
  Vec<CoriolisMatrixAux> coriolis_matrices;
  
  cudaStream_t *streams;
  cudaStream_t *streams_p2p;
  cudaStream_t *streams_energy;
  cudaEvent_t *events;

  // General device functions
  const int &n_gpus() const { return _n_gpus; }
  void setup_n_gpus();
  void devices_synchronize();
  void devices_memory_usage() const;
  void reset_devices();
  
  void setup_wavepackets_on_single_device();
  void destroy_wavepackets_on_single_device();
  
  void setup_coriolis_matrices_and_copy_to_device();
  void setup_constant_memory_on_device();
  void copy_psi_from_device_to_host();

  void dump_wavepacket();
  
  void evolution_with_potential(const double dt);
  void evolution_with_kinetic(const double dt);
  void evolution_with_rotational(const double dt);

  void evolution_with_coriolis(const double dt, const int calculate_energy = 0);

  void evolution_with_coriolis_2(const double dt, const int calculate_energy = 0);
  
  void evolution_with_coriolis_with_p2p(const double dt, const int calculate_energy = 0);
  void evolution_with_coriolis_with_p2p_async(const double dt, const int calculate_energy = 0);
  void evolution_with_coriolis_with_p2p_async_and_events(const double dt, const int calculate_energy = 0);
  void evolution_with_coriolis_with_p2p_async_and_events_2(const double dt, const int calculate_energy = 0);
  void evolution_with_coriolis_with_p2p_async_and_events_3(const double dt, const int calculate_energy = 0);
  
  void update_evolution_with_coriolis();
  
  void calculate_wavepacket_modules();
  void calculate_wavepacket_potential_energies();
  
  void calculate_kinetic_energies_for_legendre_psi();
  void calculate_rotational_energies_for_legendre_psi();
  
  void calculate_wavepacket_modules_for_legendre_psi();
  
  void forward_legendre_transform();
  void backward_legendre_transform();

  void forward_fft_for_legendre_psi();
  void backward_fft_for_legendre_psi(const int do_scale);

  void print_energies() const;
  
  void enable_peer_to_peer_access() const;
  void setup_streams_and_events(const int setup_streams, const int setup_events);
  void destroy_streams_and_events();

  void _p2p_test(const double dt = 0.0);
  void _p2p_test_2(const double dt);
};

#endif /* CUDA_OPENMP_H */
