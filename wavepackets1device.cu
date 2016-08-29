
#include <helper_cuda.h>
#include "cudaUtils.h"
#include "wavepackets1device.h"
#include "evolutionUtils.h"
#include "cudaMath.h"

#include "evolutionCUDAaux.cu"

// defined as extern in evolutionUtils.h
__constant__ EvoltionUtils::RadialCoordinate r1_dev;
__constant__ EvoltionUtils::RadialCoordinate r2_dev;
__constant__ double time_step_dev;

OmegaWavepacketsOnSingleDevice::OmegaWavepacketsOnSingleDevice(const int device_index_,
							       const int omega_start_index_,
							       const int n_omegas_,
							       const double * &pot_,
							       const RadialCoordinate &r1_,
							       const RadialCoordinate &r2_,
							       const AngleCoordinate &theta_,
							       OmegaStates &omega_states_,
							       const int &l_max_,
							       const Vec<CoriolisMatrixAux> &coriolis_matrices_) :
  _device_index(device_index_), 
  omega_start_index(omega_start_index_),
  n_omegas(n_omegas_),
  pot(pot_),
  r1(r1_), r2(r2_), theta(theta_),
  omega_states(omega_states_),
  l_max(l_max_),
  coriolis_matrices(coriolis_matrices_),
  pot_dev(0),
  work_dev(0),
  coriolis_matrices_dev(0),
  has_cublas_handle(0),
  has_cufft_plan_for_legendre_psi(0)
{
  setup_data_on_device();
}

OmegaWavepacketsOnSingleDevice::~OmegaWavepacketsOnSingleDevice()
{
  for(int i = 0; i < omega_wavepackets.size(); i++) {
    if(omega_wavepackets[i]) { delete omega_wavepackets[i]; omega_wavepackets[i] = 0; }
  }

  destroy_data_on_device();
}

int OmegaWavepacketsOnSingleDevice::current_device_index() const
{
  int dev_index = -1;
  checkCudaErrors(cudaGetDevice(&dev_index));
  return dev_index;
}

void OmegaWavepacketsOnSingleDevice::setup_device() const
{
  if(current_device_index() != device_index()) {  
    insist(device_index() >= 0);
    checkCudaErrors(cudaSetDevice(device_index()));
  }
}

void OmegaWavepacketsOnSingleDevice::setup_data_on_device()
{
  setup_device();

  insist(pot);
  
  std::cout << " setup OmegaWavepacketsOnSingleDevice on device: " << device_index() << std::endl;

  setup_constant_data_on_device();

  if(!work_dev) {
    const int max_dim = r1.n*r2.n + theta.n + 1024;
    checkCudaErrors(cudaMalloc(&work_dev, max_dim*sizeof(Complex)));
    insist(work_dev);
    std::cout << " work_dev: " << work_dev << std::endl;
  }
  
  setup_potential_on_device();
  setup_omega_wavepackets();
  setup_cublas_handle();
}

void OmegaWavepacketsOnSingleDevice::destroy_data_on_device()
{
  setup_device();
  
  std::cout << " Destroy OmegaWavepacketsOnSingleDevice on device: " << device_index() << std::endl;
  
  _CUDA_FREE_(pot_dev);
  _CUDA_FREE_(work_dev);
  _CUDA_FREE_(coriolis_matrices_dev);

  for(int i = 0; i < omega_wavepackets.size(); i++) {
    if(omega_wavepackets[i]) { 
      delete omega_wavepackets[i];
      omega_wavepackets[i] = 0;
    }
  }
  omega_wavepackets.resize(0);

  destroy_cublas_handle();
  destroy_cufft_plan_for_legendre_psi();
}

void OmegaWavepacketsOnSingleDevice::setup_potential_on_device()
{
  if(pot_dev) return;

  insist(pot);
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  checkCudaErrors(cudaMalloc(&pot_dev, n1*n2*n_theta*sizeof(double)));
  insist(pot_dev);
  checkCudaErrors(cudaMemcpyAsync(pot_dev, pot, n1*n2*n_theta*sizeof(double), cudaMemcpyHostToDevice));
}

void OmegaWavepacketsOnSingleDevice::setup_omega_wavepackets()
{
  insist(!omega_wavepackets.size());
  
  omega_wavepackets.resize(n_omegas, 0);
  
  Vec<int> omegas(n_omegas, (int *)omega_states.omegas + omega_start_index);
  
  for(int i = 0; i < n_omegas; i++) {
    omega_wavepackets[i] = new OmegaWavepacket(omegas[i], 
					       omega_states.l_max,
					       coriolis_matrices,
					       omega_states.associated_legendres[omega_start_index+i],
					       r1, r2, theta, 
					       omega_states.wave_packets[omega_start_index+i],
					       pot_dev, cublas_handle, 
					       cufft_plan_for_legendre_psi, 
					       work_dev);
    insist(omega_wavepackets[i]);
  }
}

void OmegaWavepacketsOnSingleDevice::setup_cublas_handle()
{
  if(has_cublas_handle) return;

  std::cout << " Setup CUBLAS handle on device: " << current_device_index() << std::endl;

  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  has_cublas_handle = 1;
}

void OmegaWavepacketsOnSingleDevice::destroy_cublas_handle()
{
  if(!has_cublas_handle) return;

  std::cout << " Destroy CUBLAS handle on device: " << current_device_index() << std::endl;
  
  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);
  has_cublas_handle = 0;
}

void OmegaWavepacketsOnSingleDevice::setup_constant_data_on_device()
{
#if 0
  std::cout << " Setup constane device memory on device: " << current_device_index() << std::endl;
  insist(theta.n <= 512);
  checkCudaErrors(cudaMemcpyToSymbol(gauss_legendre_weight_dev, theta.w, theta.n*sizeof(double)));
#endif
}

void OmegaWavepacketsOnSingleDevice::calculate_wavepacket_modules()
{
  setup_device();
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->calculate_wavepacket_module();
}

void OmegaWavepacketsOnSingleDevice::calculate_wavepacket_potential_energies()
{
  setup_device();
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->calculate_potential_energy();
}

void OmegaWavepacketsOnSingleDevice::calculate_wavepacket_modules_for_legendre_psi()
{
  setup_device();
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->calculate_wavepacket_module_for_legendre_psi();
}

void OmegaWavepacketsOnSingleDevice::setup_cufft_plan_for_legendre_psi()
{
  if(has_cufft_plan_for_legendre_psi) return;
  
  const int n1 = r1.n;
  const int n2 = r2.n;
  const int n_legs = l_max + 1;
  
  const int dim [] = { n2, n1 };
  
  insist(cufftPlanMany(&cufft_plan_for_legendre_psi, 2, const_cast<int *>(dim), NULL, 1,
                       n1*n2, NULL, 1, n1*n2,
                       CUFFT_Z2Z, n_legs) == CUFFT_SUCCESS);
  
  has_cufft_plan_for_legendre_psi = 1;
}

void OmegaWavepacketsOnSingleDevice::destroy_cufft_plan_for_legendre_psi()
{
  if(!has_cufft_plan_for_legendre_psi) return;
  insist(cufftDestroy(cufft_plan_for_legendre_psi) == CUFFT_SUCCESS);
  has_cufft_plan_for_legendre_psi = 0;
}

void OmegaWavepacketsOnSingleDevice::forward_legendre_transform()
{
  setup_device();
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->forward_legendre_transform();
}

void OmegaWavepacketsOnSingleDevice::backward_legendre_transform()
{
  setup_device();
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->backward_legendre_transform();
}

void OmegaWavepacketsOnSingleDevice::forward_fft_for_legendre_psi()
{
  setup_device();
  setup_cufft_plan_for_legendre_psi();
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->forward_fft_for_legendre_psi();
}

void OmegaWavepacketsOnSingleDevice::backward_fft_for_legendre_psi(const int do_scale)
{
  setup_device();
  setup_cufft_plan_for_legendre_psi();
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->backward_fft_for_legendre_psi(do_scale);
}

void OmegaWavepacketsOnSingleDevice::copy_psi_from_device_to_host()
{
  setup_device();
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->copy_psi_from_device_to_host();
}

void OmegaWavepacketsOnSingleDevice::copy_coriolis_matrices_to_device(const double *c, const int s)
{
  setup_device();

  insist(!coriolis_matrices_dev);

  checkCudaErrors(cudaMalloc(&coriolis_matrices_dev, s*sizeof(double)));
  insist(coriolis_matrices_dev);
  
  checkCudaErrors(cudaMemcpyAsync(coriolis_matrices_dev, c, s*sizeof(double), cudaMemcpyHostToDevice));
}

void OmegaWavepacketsOnSingleDevice::setup_constant_memory_on_device(const double time_step)
{
  setup_device();

  EvoltionUtils::copy_radial_coordinate_to_device(r1_dev, r1.left, r1.dr, r1.mass,
						  r1.dump_Cd, r1.dump_xd, r1.n);

  EvoltionUtils::copy_radial_coordinate_to_device(r2_dev, r2.left, r2.dr, r2.mass, 
						  r2.dump_Cd, r2.dump_xd, r2.n);
  
  checkCudaErrors(cudaMemcpyToSymbol(time_step_dev, &time_step, sizeof(double)));
}

void OmegaWavepacketsOnSingleDevice::evolution_with_potential(const double dt)
{
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->evolution_with_potential(dt);
}

void OmegaWavepacketsOnSingleDevice::evolution_with_kinetic(const double dt)
{
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->evolution_with_kinetic(dt);
}

void OmegaWavepacketsOnSingleDevice::evolution_with_rotational(const double dt)
{
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->evolution_with_rotational(dt);
}

void OmegaWavepacketsOnSingleDevice::calculate_rotational_energies_for_legendre_psi()
{
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->calculate_rotational_energy_for_legendre_psi();
}

void OmegaWavepacketsOnSingleDevice::calculate_kinetic_energies_for_legendre_psi()
{
  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->calculate_kinetic_energy_for_legendre_psi();
}

void OmegaWavepacketsOnSingleDevice::test_coriolis_matrices() const
{
  setup_device();

  std::cout << "\n" << __PRETTY_FUNCTION__ << "\n" << std::endl;

  std::cout << coriolis_matrices << std::endl;
  std::cout.flush();

  for(int i = 0; i < coriolis_matrices.size(); i++) {
    if(coriolis_matrices[i].l == -1) continue;
    std::cout << " " << coriolis_matrices[i].l << std::endl; 
    std::cout.flush();
    const int n = coriolis_matrices[i].omega_max - coriolis_matrices[i].omega_min + 1;
    const double *e = coriolis_matrices_dev + coriolis_matrices[i].offset;
    const double *v = e + n;
    _print_coriolis_on_device_<<<1, 1>>>(n, e, v);
    
    double *b_dev = 0;
    checkCudaErrors(cudaMalloc(&b_dev, n*n*sizeof(double)));
    insist(b_dev);

    _calculate_coriolis_on_device_<<<1, 1>>>(n, e, v, b_dev);
    _print_coriolis_on_device_<<<1, 1>>>(n, 0, b_dev);
    
    _CUDA_FREE_(b_dev);

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return;

  const int &n_omegas = omega_wavepackets.size();
  for(int i = 0; i < n_omegas; i++) 
    std::cout << omega_wavepackets[i]->coriolis_matrices << std::endl;
}

void OmegaWavepacketsOnSingleDevice::evolution_test(const int step, const double dt)
{
  setup_device();
  
  if(step == 0) evolution_with_potential(-dt/2);
  
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

  evolution_with_coriolis(dt/2);

  evolution_with_rotational(dt/4);

  calculate_rotational_energies_for_legendre_psi();
  calculate_coriolis_energy_for_legendre_psi();
  
  backward_legendre_transform();

  calculate_wavepacket_potential_energies();
  calculate_wavepacket_modules();
  
  std::ios_base::fmtflags old_flags = std::cout.flags();

  const int &n = omega_wavepackets.size();
  
  total_module = 0.0;
  total_energy = 0.0;
  
  for(int i = 0; i < n; i++) {
    std::cout << " " << i 
	      << std::fixed
	      << " " << omega_wavepackets[i]->wavepacket_module() 
	      << " " << omega_wavepackets[i]->potential_energy()
	      << " " << omega_wavepackets[i]->kinetic_energy()
	      << " " << omega_wavepackets[i]->rotational_energy()
	      << " " << omega_wavepackets[i]->coriolis_energy() 
	      << " " << omega_wavepackets[i]->total_energy()
	      << std::endl;
    total_module += omega_wavepackets[i]->wavepacket_module();
    total_energy += omega_wavepackets[i]->total_energy();
  }
  
  std::cout << " Total: " << total_module << " " << total_energy << std::endl;
  
  std::cout.flags(old_flags);
}

void OmegaWavepacketsOnSingleDevice::evolution_with_coriolis(const double dt)
{
  const int n_omegas = omega_wavepackets.size();
  
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->zero_psi_dev();
  
  for(int i = 0; i < n_omegas; i++) {
    for(int j = 0; j < n_omegas; j++) {
      omega_wavepackets[i]->evolution_with_coriolis(dt, omega_wavepackets[j]->omega,
						    coriolis_matrices_dev,
						    omega_wavepackets[j]->legendre_psi_dev);
    }
  }
  
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->update_evolution_with_coriolis();
}

void OmegaWavepacketsOnSingleDevice::calculate_coriolis_energy_for_legendre_psi()
{
  const int n_omegas = omega_wavepackets.size();
  
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->_coriolis_energy = 0.0;
  
  for(int i = 0; i < n_omegas; i++) {
    const int &omega_i = omega_wavepackets[i]->omega;
    for(int j = i; j < n_omegas; j++) {
      const int &omega_j = omega_wavepackets[j]->omega;
      if(omega_i == omega_j || omega_i+1 == omega_j || omega_i-1 == omega_j) {
	omega_wavepackets[i]->calculate_coriolis_energy_for_legendre_psi(omega_j, 
									 coriolis_matrices_dev,
								       omega_wavepackets[j]->legendre_psi_dev);
	
      }
    }
  }
}
