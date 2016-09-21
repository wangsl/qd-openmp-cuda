
#include <chrono>
#include <ctime>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>

#include "cudaUtils.h"
#include "cudaOpenMP.h"
#include "evolutionUtils.h"
#include "matlabData.h"
#include "cudaMath.h"

inline static void divide_into_chunks(const int n, const int m, int *chunks)
{
  memset(chunks, 0, m*sizeof(int));
  for(int i = 0; i < m; i++) chunks[i] = n/m;
  for(int i = 0; i < n-n/m*m; i++) chunks[i]++;
  int s = 0;
  for(int i = 0; i < m; i++) s += chunks[i];
  insist(s == n);
}

CudaOpenMPQMMD::CudaOpenMPQMMD() :
  _n_gpus(0), 
  streams(0), streams_p2p(0), streams_energy(0),
  events(0)
{
  setup_n_gpus();
  setup_wavepackets_on_single_device();
}

CudaOpenMPQMMD::~CudaOpenMPQMMD()
{
  std::cout << std::endl;
  destroy_wavepackets_on_single_device();
  destroy_streams_and_events();
  reset_devices();
}

void CudaOpenMPQMMD::setup_n_gpus()
{
  if(!_n_gpus) {
    checkCudaErrors(cudaGetDeviceCount(&_n_gpus));
    if(n_gpus() == 1) 
      std::cout << " There is 1 GPU card" << std::endl;
    else
      std::cout << " There are " << n_gpus() << " GPU cards" << std::endl;
  }
}

void CudaOpenMPQMMD::test()
{
  insist(n_gpus() == 1);

  const int &total_steps = MatlabData::time()->total_steps;
  int &steps = MatlabData::time()->steps;
  const double &dt = MatlabData::time()->time_step;

  for(int L = 0; L < total_steps; L++) {
    
    std::cout << std::endl << " Step: " << steps << ", " << time_now() << std::endl;
    
    omp_set_num_threads(n_gpus());
    
    for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
      omega_wavepackets_on_single_device[i_dev]->evolution_test(L, dt);
    
    devices_synchronize();
    
    steps++;
    
    if(MatlabData::options()->wave_to_matlab &&
       steps%MatlabData::options()->steps_to_copy_psi_from_device_to_host == 0) {
      copy_psi_from_device_to_host();
      wavepacket_to_matlab(MatlabData::options()->wave_to_matlab);
    }
  }
}

void CudaOpenMPQMMD::setup_wavepackets_on_single_device()
{
  insist(n_gpus() > 0);
  
  insist(omega_wavepackets_on_single_device.size() == 0);
  omega_wavepackets_on_single_device.resize(n_gpus(), 0);
  
  const int &n = omega_wavepackets_on_single_device.size();
  
  Vec<int> omegas_index(n);
  divide_into_chunks(MatlabData::omega_states()->omegas.size(), n, omegas_index);
  
  std::cout << " Omegas index: ";
  omegas_index.show_in_one_line();
  
  int omega_start_index = 0;
  for(int i_dev = 0; i_dev < n; i_dev++) {
    
    const int n_omegas = omegas_index[i_dev];

    checkCudaErrors(cudaSetDevice(i_dev));
    
    omega_wavepackets_on_single_device[i_dev] = 
      new OmegaWavepacketsOnSingleDevice(i_dev, omega_start_index,
					 n_omegas, coriolis_matrices);
    
    insist(omega_wavepackets_on_single_device[i_dev]);
    
    omega_start_index += n_omegas;
  }
  
  // OmegaWavepacketsOnSingleDevice::coriolis_matrices_dev are setup here
  // reference to double *, OmegaWavepacket uses the same reference
  // so we can not set it as const double * &coriolis_matrices_dev;
  setup_coriolis_matrices_and_copy_to_device();
  
  setup_constant_memory_on_device();

  if(n_gpus() > 1) {
    for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
      omega_wavepackets_on_single_device[i_dev]->setup_work_dev_2();
    }
    enable_peer_to_peer_access();
  }

  devices_synchronize();
  devices_memory_usage();
}

void CudaOpenMPQMMD::destroy_wavepackets_on_single_device()
{
  const int &n = omega_wavepackets_on_single_device.size();
  for(int i = 0; i < n; i++) {
    if(omega_wavepackets_on_single_device[i]) {
      delete omega_wavepackets_on_single_device[i];
      omega_wavepackets_on_single_device[i] = 0; 
    }
  }
  omega_wavepackets_on_single_device.resize(0);
}

void CudaOpenMPQMMD::devices_synchronize()
{
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

void CudaOpenMPQMMD::devices_memory_usage() const
{
  std::cout << std::endl;
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    cudaUtils::gpu_memory_usage();
  }
}

void CudaOpenMPQMMD::setup_coriolis_matrices_and_copy_to_device()
{
  std::cout << " Setup Coriolis Matrices and copy to devices" << std::endl;
  
  const int &l_max = MatlabData::omega_states()->l_max;
  
  coriolis_matrices.resize(l_max+1);
  
  const int &omega_min = MatlabData::omega_states()->omegas[0];
  
  int s = 0;
  for(int l = omega_min; l < l_max+1; l++) {
    
    coriolis_matrices[l].offset = s;
    
    coriolis_matrices[l].l = l;
    
    calculate_coriolis_matrix_dimension(MatlabData::omega_states()->J,
					MatlabData::omega_states()->parity,
					coriolis_matrices[l].l,
					coriolis_matrices[l].omega_min,
					coriolis_matrices[l].omega_max);
    
    const int n = coriolis_matrices[l].omega_max - coriolis_matrices[l].omega_min + 1;
    
    s += n*(n+1);
  }

  double *cor_mats = new double [s];
  insist(cor_mats);
  memset(cor_mats, 0, s*sizeof(double));
  
  for(int l = 0; l < l_max+1; l++) {
    
    if(coriolis_matrices[l].l == -1) continue;
    
    const int &omega_min = coriolis_matrices[l].omega_min;
    const int &omega_max = coriolis_matrices[l].omega_max;
    const int n = omega_max - omega_min + 1;
    RMat cor_mat_(n, n+1, cor_mats+coriolis_matrices[l].offset);
    
    setup_coriolis_matrix(MatlabData::omega_states()->J,
			  MatlabData::omega_states()->parity,
			  coriolis_matrices[l].l, cor_mat_);
  }
  
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    omega_wavepackets_on_single_device[i_dev]->copy_coriolis_matrices_to_device(cor_mats, s);
  }
  
  if(cor_mats) { delete [] cor_mats; cor_mats = 0; }
}

void CudaOpenMPQMMD::setup_constant_memory_on_device()
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    omega_wavepackets_on_single_device[i_dev]->setup_constant_memory_on_device();
  }
}

void CudaOpenMPQMMD::reset_devices()
{
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    checkCudaErrors(cudaDeviceReset());
  }
}

void CudaOpenMPQMMD::test_coriolis() const
{ }

void CudaOpenMPQMMD::copy_psi_from_device_to_host()
{
  std::cout << " Copy wavepacket data from devices to host" << std::endl;
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    omega_wavepackets_on_single_device[i_dev]->copy_psi_from_device_to_host();
  }
  devices_synchronize();
}

void CudaOpenMPQMMD::evolution_with_potential(const double dt)
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->evolution_with_potential(dt);
}

void CudaOpenMPQMMD::evolution_with_kinetic(const double dt)
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->evolution_with_kinetic(dt);
}

void CudaOpenMPQMMD::evolution_with_rotational(const double dt)
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->evolution_with_rotational(dt);
}

void CudaOpenMPQMMD::calculate_wavepacket_modules()
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->calculate_wavepacket_modules();
}

void CudaOpenMPQMMD::calculate_wavepacket_potential_energies()
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->calculate_wavepacket_potential_energies();
}

void CudaOpenMPQMMD::calculate_kinetic_energies_for_legendre_psi()
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->calculate_kinetic_energies_for_legendre_psi();
}

void CudaOpenMPQMMD::calculate_rotational_energies_for_legendre_psi()
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->calculate_rotational_energies_for_legendre_psi();
}

void CudaOpenMPQMMD::forward_legendre_transform()
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->forward_legendre_transform();
}

void CudaOpenMPQMMD::backward_legendre_transform()
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->backward_legendre_transform();
}

void CudaOpenMPQMMD::forward_fft_for_legendre_psi()
{						
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->forward_fft_for_legendre_psi();
}

void CudaOpenMPQMMD::backward_fft_for_legendre_psi(const int do_scale)
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->backward_fft_for_legendre_psi(do_scale);
}

void CudaOpenMPQMMD::dump_wavepacket()
{
  if(!MatlabData::dump_wavepacket()) return;
  
  std::cout << " Dump wavepacket" << std::endl;

#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->dump_wavepacket();
}

void CudaOpenMPQMMD::test_multiple_cards()
{
  const int &total_steps = MatlabData::time()->total_steps;
  int &steps = MatlabData::time()->steps;
  const double &dt = MatlabData::time()->time_step;

  omp_set_num_threads(n_gpus());

  checkCudaErrors(cudaProfilerStart());
  
  for(int L = 0; L < total_steps; L++) {
    
    std::cout << std::endl << " Step: " << steps << ", " << time_now() << std::endl;

    dump_wavepacket();
    
    if(steps == 0) evolution_with_potential(-dt/2);
    
    evolution_with_potential(dt);
    
    forward_legendre_transform();
    
    evolution_with_rotational(dt/4);
    
    evolution_with_coriolis(dt/2);
    
    evolution_with_rotational(dt/4);
    
    forward_fft_for_legendre_psi();
    
    evolution_with_kinetic(dt);
    
    calculate_kinetic_energies_for_legendre_psi();
    
    backward_fft_for_legendre_psi(1);
    
    evolution_with_rotational(dt/4);
    
    evolution_with_coriolis(dt/2, 1);
    
    evolution_with_rotational(dt/4);
    
    calculate_rotational_energies_for_legendre_psi();
    
    backward_legendre_transform();
    
    calculate_wavepacket_potential_energies();
    
    calculate_wavepacket_modules();

    print_energies();

    if(MatlabData::options()->wave_to_matlab && 
       steps%MatlabData::options()->steps_to_copy_psi_from_device_to_host == 0) {
      copy_psi_from_device_to_host();
      wavepacket_to_matlab(MatlabData::options()->wave_to_matlab);
    }

    steps++;
  }
  
  checkCudaErrors(cudaProfilerStop());
}

void CudaOpenMPQMMD::evolution_with_coriolis(const double dt, const int calculate_energy)
{

  //evolution_with_coriolis_2(dt, calculate_energy);  return;

#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    
    omega_wavepackets_on_single_device[i_dev]->zero_coriolis_variables();
    
    omega_wavepackets_on_single_device[i_dev]->evolution_with_coriolis_on_same_device(dt);
    
    if(calculate_energy)
      omega_wavepackets_on_single_device[i_dev]->calculate_coriolis_energy_for_legendre_psi();
  }
  
  if(n_gpus() > 1) {
    if(MatlabData::options()->use_p2p_async == 1) {
      evolution_with_coriolis_with_p2p_async(dt, calculate_energy);
    } else if(MatlabData::options()->use_p2p_async == 2) {
      evolution_with_coriolis_with_p2p_async_and_events(dt, calculate_energy);
    } else if(MatlabData::options()->use_p2p_async == 3) {
      evolution_with_coriolis_with_p2p_async_and_events_2(dt, calculate_energy);
    } else if(MatlabData::options()->use_p2p_async == 4) {
      evolution_with_coriolis_with_p2p_async_and_events_3(dt, calculate_energy);
    } else {
      evolution_with_coriolis_with_p2p(dt, calculate_energy);
    }
  }
  
  update_evolution_with_coriolis();
}

void CudaOpenMPQMMD::print_energies() const
{
  std::ios_base::fmtflags old_flags = std::cout.flags();
  
  double total_module = 0.0;
  double total_energy = 0.0;
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    omega_wavepackets_on_single_device[i_dev]->print_energies();
    total_module += omega_wavepackets_on_single_device[i_dev]->total_module;
    total_energy += omega_wavepackets_on_single_device[i_dev]->total_energy;
  }

  if(total_module > 1.0) {
    std::cout << " *** Total module error: " << total_module << " ***" << std::endl;
    insist(total_module <= 1);
  }
  
  std::cout << " Total: " << total_module << " " << total_energy << std::endl;
  
  std::cout.flags(old_flags);
}

void CudaOpenMPQMMD::update_evolution_with_coriolis()
{
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) 
    omega_wavepackets_on_single_device[i_dev]->update_evolution_with_coriolis();
}

void CudaOpenMPQMMD::enable_peer_to_peer_access() const
{
  if(n_gpus() == 1) return;
  
  std::cout << " Enable peer to peer access" << std::endl;
  
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    for(int j_dev = i_dev+1; j_dev < n_gpus(); j_dev++) {
      checkCudaErrors(cudaSetDevice(i_dev));
      checkCudaErrors(cudaDeviceEnablePeerAccess(j_dev, 0));
      
      checkCudaErrors(cudaSetDevice(j_dev));
      checkCudaErrors(cudaDeviceEnablePeerAccess(i_dev, 0));
    }
  }
}

void CudaOpenMPQMMD::setup_streams_and_events(const int setup_streams, const int setup_events)
{
  if(n_gpus() == 1) return;
  
  if(streams) return;
  
  if(setup_events) insist(setup_streams);

  if(setup_streams && setup_events)
    std::cout << " Setup streams and events" << std::endl;
  else if(setup_streams)
    std::cout << " Setup streams" << std::endl;
  
  if(!streams && setup_streams) {
    streams = (cudaStream_t *) malloc(n_gpus()*sizeof(cudaStream_t));
    insist(streams);
    
    streams_p2p = (cudaStream_t *) malloc(n_gpus()*sizeof(cudaStream_t));
    insist(streams_p2p);

    streams_energy = (cudaStream_t *) malloc(n_gpus()*sizeof(cudaStream_t));
    insist(streams_energy);
    
    for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
      checkCudaErrors(cudaSetDevice(i_dev));
      checkCudaErrors(cudaStreamCreate(&streams[i_dev]));
      checkCudaErrors(cudaStreamCreate(&streams_p2p[i_dev]));
      checkCudaErrors(cudaStreamCreate(&streams_energy[i_dev]));
    }
  }
  
  if(!events && setup_streams && setup_events) {
    events = (cudaEvent_t *) malloc(n_gpus()*sizeof(cudaEvent_t));
    insist(events);
    for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
      checkCudaErrors(cudaSetDevice(i_dev));
      checkCudaErrors(cudaEventCreateWithFlags(&events[i_dev], cudaEventDisableTiming));
    }
  }
}

void CudaOpenMPQMMD::destroy_streams_and_events()
{
  if(!streams && !events) return;

  std::cout << " Destroy streams and events" << std::endl;
  
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    if(streams) checkCudaErrors(cudaStreamDestroy(streams[i_dev]));
    if(streams_p2p) checkCudaErrors(cudaStreamDestroy(streams_p2p[i_dev]));
    if(streams_energy) checkCudaErrors(cudaStreamDestroy(streams_energy[i_dev]));
    if(events) checkCudaErrors(cudaEventDestroy(events[i_dev]));
  }
  if(streams) { free(streams); streams = 0; }
  if(streams_p2p) { free(streams_p2p); streams_p2p = 0; }
  if(streams_energy) { free(streams_energy); streams_energy = 0; }
  if(events) { free(events); events = 0; }
  
  for(int i_dev = 0; i_dev < n_gpus(); i_dev++) {
    for(int j_dev = i_dev+1; j_dev < n_gpus(); j_dev++) {
      checkCudaErrors(cudaSetDevice(i_dev));
      checkCudaErrors(cudaDeviceDisablePeerAccess(j_dev));
      
      checkCudaErrors(cudaSetDevice(j_dev));
      checkCudaErrors(cudaDeviceDisablePeerAccess(i_dev));
    }
  }
}
