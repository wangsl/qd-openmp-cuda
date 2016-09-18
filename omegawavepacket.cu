
#include <helper_cuda.h>
#include "omegawavepacket.h"
#include "deviceConst.h"
#include "cudaUtils.h"
#include "cudaMath.h"
#include "matlabData.h"

#include "evolutionCUDAaux.cu"

/*******
 * legendre_psi_dev dimesnions 0..l_max (lmax+1)
 * real data is legendre_psi_dev(:, :, omega..lmax)
 * but for FFT, we'll have to use all data in order to use same FFT plan
 ******/

OmegaWavepacket::OmegaWavepacket(const int &omega_,
				 const Vec<CoriolisMatrixAux> &coriolis_matrices_,
				 const RMat &associated_legendres_,
				 Complex *psi_, 
				 const double *pot_dev_,
				 double * &coriolis_matrices_dev_,
				 cublasHandle_t &cublas_handle_,
				 cufftHandle &cufft_plan_for_legendre_psi_,
				 Complex * &work_dev_) :
  omega(omega_), 
  l_max(MatlabData::omega_states()->l_max),
  coriolis_matrices(coriolis_matrices_),
  associated_legendres(associated_legendres_),
  r1(*MatlabData::r1()), 
  r2(*MatlabData::r2()),
  theta(*MatlabData::theta()),
  psi(psi_), 
  pot_dev(pot_dev_), 
  coriolis_matrices_dev(coriolis_matrices_dev_),
  psi_dev(0),
  legendre_psi_dev(0), associated_legendres_dev(0), 
  weighted_associated_legendres_dev(0),
  cublas_handle(cublas_handle_),
  cufft_plan_for_legendre_psi(cufft_plan_for_legendre_psi_),
  work_dev(work_dev_),
  _wavepacket_module(0), _potential_energy(0), _kinetic_energy(0), 
  _rotational_energy(0), _coriolis_energy(0),
  _wavepacket_module_for_legendre_psi(0)
{ 
  insist(theta.n > l_max+1);

  insist(psi);
  insist(work_dev);
  setup_device_data();
}

OmegaWavepacket::~OmegaWavepacket() 
{
  std::cout << " Destruct OmegaWavepacket: omega = " << omega << std::endl;
  psi = 0;
  pot_dev = 0;
  coriolis_matrices_dev = 0;
  
  _CUDA_FREE_(psi_dev);
  _CUDA_FREE_(legendre_psi_dev);
  _CUDA_FREE_(associated_legendres_dev);
  _CUDA_FREE_(weighted_associated_legendres_dev);
}

void OmegaWavepacket::setup_device_data()
{
  std::cout << " Setup OmegaWavepacket: omega = " << omega << " l_max = "<< l_max << std::endl;
  
  copy_psi_from_host_to_device();
  setup_associated_legendres();
  setup_weighted_associated_legendres();
  setup_legendre_psi();
}

void OmegaWavepacket::calculate_wavepacket_module()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  const double *w = theta.w;

  double &sum = _wavepacket_module;
  
  sum = 0.0;
  for(int k = 0; k < n_theta; k++) {
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, 
		       (const cuDoubleComplex *) psi_dev+k*n1*n2, 1,
		       (const cuDoubleComplex *) psi_dev+k*n1*n2, 1,
		       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);

    sum += w[k]*dot.real();
  }
  
  sum *= r1.dr*r2.dr;
}

void OmegaWavepacket::calculate_potential_energy()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  const double *w = theta.w;
  
  insist(work_dev);
  Complex *psi_tmp = work_dev;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);
  
  double &sum = _potential_energy;
  sum = 0.0;
  for(int k = 0; k < n_theta; k++) {
    
    cudaMath::_vector_multiplication_<Complex, Complex, double><<<n_blocks, n_threads>>>
      (psi_tmp, psi_dev+k*n1*n2, pot_dev+k*n1*n2, n1*n2);
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, 
		       (const cuDoubleComplex *) psi_dev+k*n1*n2, 1, 
		       (const cuDoubleComplex *) psi_tmp, 1, 
		       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    sum += w[k]*dot.real();
  }
  
  sum *= r1.dr*r2.dr;
}

void OmegaWavepacket::setup_associated_legendres()
{
  if(associated_legendres_dev) return;
  
  const int &n_theta = theta.n;
  const int n_legs = l_max - omega + 1;

  const RMat &p = associated_legendres;
  insist(p.rows() == n_theta && p.columns() == n_legs);
  
  Mat<Complex> p_complex(n_legs, n_theta);
  for(int l = 0; l < n_legs; l++) {
    for(int k = 0; k < n_theta; k++) {
      p_complex(l,k) = Complex(p(k,l), 0.0);
    }
  }
  
  const size_t size = n_legs*n_theta;
  checkCudaErrors(cudaMalloc(&associated_legendres_dev, size*sizeof(Complex)));
  checkCudaErrors(cudaMemcpyAsync(associated_legendres_dev, (const Complex *) p_complex,
				  size*sizeof(Complex), cudaMemcpyHostToDevice));
}

void OmegaWavepacket::setup_weighted_associated_legendres()
{
  if(weighted_associated_legendres_dev) return;
  
  const int &n_theta = theta.n;
  const int n_legs = l_max - omega + 1;
  
  const double *w = theta.w;

  const RMat &p = associated_legendres;
  insist(p.rows() == n_theta && p.columns() == n_legs);

  Mat<Complex> wp_complex(n_theta, n_legs);
  for(int l = 0; l < n_legs; l++) {
    for(int k = 0; k < n_theta; k++) {
      wp_complex(k,l) = Complex(w[k]*p(k,l), 0.0);
    }
  }
  
  const size_t size = n_theta*n_legs;
  checkCudaErrors(cudaMalloc(&weighted_associated_legendres_dev, size*sizeof(Complex)));
  checkCudaErrors(cudaMemcpyAsync(weighted_associated_legendres_dev, 
				  (const Complex *) wp_complex,
				  size*sizeof(Complex), cudaMemcpyHostToDevice));
}

void OmegaWavepacket::setup_legendre_psi()
{
  if(legendre_psi_dev) return;
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max + 1;
  
  std::cout << " Allocate device memory for Legendre psi: "
            << n1 << " " << n2 << " " << n_legs << std::endl;
  
  const size_t size = n1*n2*n_legs;
  checkCudaErrors(cudaMalloc(&legendre_psi_dev, size*sizeof(Complex)));
  insist(legendre_psi_dev);
  checkCudaErrors(cudaMemset(legendre_psi_dev, 0, size*sizeof(Complex)));
}

void OmegaWavepacket::forward_legendre_transform()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int n_legs = l_max - omega + 1;
  
  const Complex one(1.0, 0.0);
  const Complex zero(0.0, 0.0);

  Complex *legendre_psi_dev_ = legendre_psi_dev + omega*n1*n2;
  
  insist(cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     n1*n2, n_legs, n_theta, 
                     (const cuDoubleComplex *) &one,
                     (const cuDoubleComplex *) psi_dev, n1*n2,
                     (const cuDoubleComplex *) weighted_associated_legendres_dev, n_theta,
                     (const cuDoubleComplex *) &zero,
                     (cuDoubleComplex *) legendre_psi_dev_, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void OmegaWavepacket::backward_legendre_transform()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int n_legs = l_max - omega + 1;
  
  const Complex one(1.0, 0.0);
  const Complex zero(0.0, 0.0);
  
  const Complex *legendre_psi_dev_ = legendre_psi_dev + omega*n1*n2;

  insist(cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     n1*n2, n_theta, n_legs,
                     (const cuDoubleComplex *) &one,
                     (const cuDoubleComplex *) legendre_psi_dev_, n1*n2,
                     (const cuDoubleComplex *) associated_legendres_dev, n_legs,
                     (const cuDoubleComplex *) &zero,
                     (cuDoubleComplex *) psi_dev, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void OmegaWavepacket::forward_fft_for_legendre_psi()
{ 
  insist(cufftExecZ2Z(cufft_plan_for_legendre_psi, 
		      (cuDoubleComplex *) legendre_psi_dev,
                      (cuDoubleComplex *) legendre_psi_dev, 
		      CUFFT_FORWARD) == CUFFT_SUCCESS);
}

void OmegaWavepacket::backward_fft_for_legendre_psi(const int do_scale)
{
  insist(cufftExecZ2Z(cufft_plan_for_legendre_psi, 
		      (cuDoubleComplex *) legendre_psi_dev, 
                      (cuDoubleComplex *) legendre_psi_dev, 
		      CUFFT_INVERSE) == CUFFT_SUCCESS);
  
  if(do_scale) {
    const int &n1 = r1.n;
    const int &n2 = r2.n;
    const int n_legs = l_max - omega + 1;

    Complex *legendre_psi_dev_ = legendre_psi_dev + omega*n1*n2;
    
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(cublas_handle, n1*n2*n_legs, &s, 
			(cuDoubleComplex *) legendre_psi_dev_, 1) 
           == CUBLAS_STATUS_SUCCESS);
  }
}

void OmegaWavepacket::copy_psi_from_device_to_host()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  insist(psi && psi_dev);
  checkCudaErrors(cudaMemcpy(psi, psi_dev, n1*n2*n_theta*sizeof(Complex), 
			     cudaMemcpyDeviceToHost));
}

void OmegaWavepacket::copy_psi_from_host_to_device()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  if(!psi_dev) 
    checkCudaErrors(cudaMalloc(&psi_dev, n1*n2*n_theta*sizeof(Complex)));
  insist(psi_dev);
  
  checkCudaErrors(cudaMemcpyAsync(psi_dev, psi, n1*n2*n_theta*sizeof(Complex), 
				  cudaMemcpyHostToDevice));
}

void OmegaWavepacket::calculate_wavepacket_module_for_legendre_psi()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;
  
  const Complex *legendre_psi_dev_ = legendre_psi_dev + n1*n2*omega;
  
  Complex s(0.0, 0.0);
  
  insist(cublasZdotc(cublas_handle, n1*n2*n_legs,
		     (const cuDoubleComplex *) legendre_psi_dev_, 1, 
		     (const cuDoubleComplex *) legendre_psi_dev_, 1,
		     (cuDoubleComplex *) &s) == CUBLAS_STATUS_SUCCESS);
  
  _wavepacket_module_for_legendre_psi = s.real()*r1.dr*r2.dr;
}

void OmegaWavepacket::evolution_with_potential(const double dt)
{
  insist(pot_dev && psi_dev);
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  const size_t n = n1*n2*n_theta;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
  
  _evolution_with_potential_<<<n_blocks, n_threads>>>(psi_dev, pot_dev, n, dt);
}

void OmegaWavepacket::evolution_with_kinetic(const double dt)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;
  
  const size_t n = n1*n2*n_legs;

  Complex *legendre_psi_dev_ = legendre_psi_dev + n1*n2*omega;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
  
  _evolution_with_kinetic_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
    (legendre_psi_dev_, n1, n2, n_legs, dt);
}

void OmegaWavepacket::evolution_with_rotational(const double dt)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;

  const size_t n = n1*n2*n_legs;

  Complex *legendre_psi_dev_ = legendre_psi_dev + n1*n2*omega;

  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
  
  _evolution_with_rotational_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
    (legendre_psi_dev_, n1, n2, n_legs, omega, dt);
}

void OmegaWavepacket::calculate_kinetic_energy_for_legendre_psi()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  
  insist(work_dev);
  Complex *psi_tmp_dev = work_dev;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);
  
  double &sum = _kinetic_energy;
  
  sum = 0.0;
  for(int l = omega; l < l_max+1; l++) {
    
    const Complex *legendre_psi_dev_ = legendre_psi_dev + n1*n2*l;
    
    _psi_times_kinitic_energy_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
      (psi_tmp_dev, legendre_psi_dev_, n1, n2);
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, 
		       (const cuDoubleComplex *) legendre_psi_dev_, 1, 
		       (const cuDoubleComplex *) psi_tmp_dev, 1, 
                       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    sum += dot.real();
  }
  
  sum *= r1.dr*r2.dr/(n1*n2);
}

void OmegaWavepacket::calculate_rotational_energy_for_legendre_psi()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  
  insist(work_dev);
  Complex *psi_tmp_dev = work_dev;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);
  
  double &sum = _rotational_energy;
  
  sum = 0.0;
  for(int l = omega; l < l_max+1; l++) {

    const Complex *legendre_psi_dev_ = legendre_psi_dev + n1*n2*l;
    
    _psi_times_moments_of_inertia_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
      (psi_tmp_dev, legendre_psi_dev_, n1, n2);
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, 
		       (const cuDoubleComplex *) legendre_psi_dev_, 1, 
		       (const cuDoubleComplex *) psi_tmp_dev, 1, 
                       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    sum += l*(l+1)*dot.real();
  }
  
  sum *= r1.dr*r2.dr;
}

void OmegaWavepacket::evolution_with_coriolis(const double dt, 
					      const int l, const int omega1,
					      const Complex *legendre_psi_omega1,
					      cudaStream_t *stream)
{
  insist(coriolis_matrices_dev);
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  
  Complex *legendre_psi_dev_ = psi_dev + l*n1*n2;
  
  const int n = coriolis_matrices[l].omega_max - coriolis_matrices[l].omega_min + 1;
  const double *e = coriolis_matrices_dev + coriolis_matrices[l].offset;
  const double *v = e + n;

  const int &omega_min = coriolis_matrices[l].omega_min;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);
  
  if(stream) {
    _evolution_with_coriolis_<<<n_blocks, n_threads, n1*sizeof(Complex), *stream>>>
      (legendre_psi_dev_, n1, n2, e, v, n, 
       omega-omega_min, omega1-omega_min, 
       dt, legendre_psi_omega1);
  } else {
    _evolution_with_coriolis_<<<n_blocks, n_threads, n1*sizeof(Complex)>>>
      (legendre_psi_dev_, n1, n2, e, v, n, 
       omega-omega_min, omega1-omega_min, 
       dt, legendre_psi_omega1);
  }
}

void OmegaWavepacket::evolution_with_coriolis(const double dt, const int omega1,
					      const Complex *legendre_psi_omega1,
					      cudaStream_t *stream)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  
  for(int l = 0; l < l_max+1; l++) {
    
    if(coriolis_matrices[l].l == -1) continue;
    
    evolution_with_coriolis(dt, l, omega1, legendre_psi_omega1+l*n1*n2, stream);
  }
}

void OmegaWavepacket::update_evolution_with_coriolis()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;
  
  checkCudaErrors(cudaMemcpy(legendre_psi_dev + n1*n2*omega, 
			     psi_dev + n1*n2*omega, 
			     n1*n2*n_legs*sizeof(Complex), 
			     cudaMemcpyDeviceToDevice));
}

void OmegaWavepacket::dump_wavepacket()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2*n_theta);
  
  _dump_wavepacket_<<<n_blocks, n_threads>>>(psi_dev, n1, n2, n_theta);
}

void OmegaWavepacket::calculate_coriolis_energy_for_legendre_psi(const int omega1,
								 const Complex *legendre_psi_omega1,
								 cudaStream_t *stream)
{
  if(omega > omega1) return;

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  
  insist(work_dev);
  Complex *psi_tmp = work_dev;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2); 
  
  if(stream) insist(cublasSetStream(cublas_handle, *stream) == CUBLAS_STATUS_SUCCESS);
  
  for(int l = 0; l < l_max+1; l++) {
    
    if(coriolis_matrices[l].l == -1) continue;
    
    const int n = coriolis_matrices[l].omega_max - coriolis_matrices[l].omega_min + 1;
    const double *e = coriolis_matrices_dev + coriolis_matrices[l].offset;
    const double *v = e + n;
    
    const int &omega_min = coriolis_matrices[l].omega_min;
    
    if(stream) {
      _coriolis_matrices_production_<<<n_blocks, n_threads, n1*sizeof(double), *stream>>>
	(psi_tmp, legendre_psi_omega1+l*n1*n2, n1, n2, e, v, n, 
	 omega-omega_min, omega1-omega_min);
    } else {
      _coriolis_matrices_production_<<<n_blocks, n_threads, n1*sizeof(double)>>>
	(psi_tmp, legendre_psi_omega1+l*n1*n2, n1, n2, e, v, n, 
	 omega-omega_min, omega1-omega_min);
    }

    Complex dot(0.0, 0.0);
    
    insist(cublasZdotc(cublas_handle, n1*n2, 
		       (const cuDoubleComplex *) legendre_psi_dev+l*n1*n2, 1, 
		       (const cuDoubleComplex *) psi_tmp, 1, 
		       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    if(omega != omega1) dot *= 2;
    
    _coriolis_energy += dot.real()*r1.dr*r2.dr;
  }

  if(stream) insist(cublasSetStream(cublas_handle, NULL) == CUBLAS_STATUS_SUCCESS);
}

void OmegaWavepacket::zero_coriolis_variables()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const size_t size = n1*n2*n_theta;
  checkCudaErrors(cudaMemset(psi_dev, 0, size*sizeof(Complex)));
  
  _coriolis_energy = 0.0;
}
