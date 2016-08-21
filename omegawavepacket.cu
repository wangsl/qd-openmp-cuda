
#include <helper_cuda.h>
#include "omegawavepacket.h"
#include "deviceConst.h"
#include "cudaUtils.h"
#include "cudaMath.h"

#include "evolutionCUDAaux.cu"

/*
  legendre_psi_dev dimesnions 0..l_max (lmax+1)
  real data is legendre_psi_dev(:, :, omega..lmax)
  but for FFT, we'll have to use all data
*/

OmegaWavepacket::OmegaWavepacket(const int &omega_,
				 const int &l_max_,
				 const Vec<CoriolisMatrixAux> &coriolis_matrices_,
				 const RMat &associated_legendres_,
				 const RadialCoordinate &r1_,
				 const RadialCoordinate &r2_,
				 const AngleCoordinate &theta_,
				 Complex *psi_, 
				 const double *pot_dev_,
				 cublasHandle_t &cublas_handle_,
				 cufftHandle &cufft_plan_for_legendre_psi_,
				 Complex * &work_dev_) :
  omega(omega_), l_max(l_max_), coriolis_matrices(coriolis_matrices_),
  associated_legendres(associated_legendres_),
  r1(r1_), r2(r2_), theta(theta_),
  psi(psi_), 
  pot_dev(pot_dev_), psi_dev(0),
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
  
  cudaStream_t *streams = (cudaStream_t *) malloc(n_theta*sizeof(cudaStream_t));
  insist(streams);
  for(int k = 0; k < n_theta; k++) 
    checkCudaErrors(cudaStreamCreate(&streams[k]));
  
  Complex *dots = new Complex [n_theta];
  insist(dots);
  memset(dots, 0, sizeof(Complex)*n_theta);
  
  const cuDoubleComplex *psi_ = (cuDoubleComplex *) psi_dev;
  
  for(int k = 0; k < n_theta; k++) {
    insist(cublasSetStream(cublas_handle, streams[k]) == CUBLAS_STATUS_SUCCESS);

    insist(cublasZdotc(cublas_handle, n1*n2, psi_, 1, psi_, 1, (cuDoubleComplex *) &dots[k]) ==
	   CUBLAS_STATUS_SUCCESS);
    
    psi_ += n1*n2;
  }
  
  checkCudaErrors(cudaDeviceSynchronize());
  
  double &sum = _wavepacket_module;
  sum = 0.0;
  for(int k = 0; k < n_theta; k++)
    sum += w[k]*dots[k].real();
  
  sum *= r1.dr*r2.dr;
  
  if(dots) { delete [] dots; dots = 0; }
  
  for(int k = 0; k < n_theta; k++) 
    checkCudaErrors(cudaStreamDestroy(streams[k]));
  
  if(streams) { free(streams); streams = 0; }
  
  cublasSetStream(cublas_handle, 0);
}

void OmegaWavepacket::calculate_potential_energy()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  const double *w = theta.w;
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_ = (cuDoubleComplex *) work_dev;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);
  
  double &sum = _potential_energy;
  sum = 0.0;
  for(int k = 0; k < n_theta; k++) {
    
    const cuDoubleComplex *psi_ = (cuDoubleComplex *) (psi_dev + k*n1*n2);
    
    cudaMath::_vector_multiplication_<Complex, Complex, double><<<n_blocks, n_threads>>>
      ((Complex *) psi_tmp_, (const Complex *) psi_, pot_dev+k*n1*n2, n1*n2);
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, psi_, 1, psi_tmp_, 1, (cuDoubleComplex *) &dot) ==
	   CUBLAS_STATUS_SUCCESS);
    
    sum += w[k]*dot.real();
  }
  
  sum *= r1.dr*r2.dr;
}

void OmegaWavepacket::setup_associated_legendres()
{
  if(associated_legendres_dev) return;
  
  const int &n_theta = theta.n;
  const int n_legs = l_max - omega + 1;
  insist(n_legs > 0);
  
  const RMat &p = associated_legendres;
  insist(p.rows() == n_theta);
  
  Mat<Complex> p_complex(n_legs, n_theta);
  for(int l = 0; l < n_legs; l++) {
    for(int k = 0; k < n_theta; k++) {
      p_complex(l,k) = Complex(p(k,l), 0.0);
    }
  }
  
  std::cout << " Allocate device memory for complex associated Legendre Polynomials: " 
            << n_legs << " " << n_theta << std::endl;
  
  const int size = n_legs*n_theta;
  checkCudaErrors(cudaMalloc(&associated_legendres_dev, size*sizeof(Complex)));
  checkCudaErrors(cudaMemcpyAsync(associated_legendres_dev, (const Complex *) p_complex,
				  size*sizeof(Complex), cudaMemcpyHostToDevice));
}

void OmegaWavepacket::setup_weighted_associated_legendres()
{
  if(weighted_associated_legendres_dev) return;
  
  const int &n_theta = theta.n;
  const int n_legs = l_max - omega + 1;
  insist(n_legs > 0);
  
  const double *w = theta.w;
  
  const RMat &p = associated_legendres;
  insist(p.rows() == n_theta);
  
  Mat<Complex> wp_complex(n_theta, n_legs);
  for(int l = 0; l < n_legs; l++) {
    for(int k = 0; k < n_theta; k++) {
      wp_complex(k,l) = Complex(w[k]*p(k,l), 0.0);
    }
  }
  
  std::cout << " Allocate device memory for weighted complex associated Legendre Polynomials: " 
            << n_theta << " " << n_legs << std::endl;
  
  const int size = n_theta*n_legs;
  checkCudaErrors(cudaMalloc(&weighted_associated_legendres_dev, size*sizeof(Complex)));
  checkCudaErrors(cudaMemcpyAsync(weighted_associated_legendres_dev, (const Complex *) wp_complex,
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

  Complex *legendre_psi_dev_ = legendre_psi_dev + omega*n1*n2;

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
  insist(cufftExecZ2Z(cufft_plan_for_legendre_psi, (cuDoubleComplex *) legendre_psi_dev,
                      (cuDoubleComplex *) legendre_psi_dev, CUFFT_FORWARD) == CUFFT_SUCCESS);
}

void OmegaWavepacket::backward_fft_for_legendre_psi(const int do_scale)
{
  insist(cufftExecZ2Z(cufft_plan_for_legendre_psi, (cuDoubleComplex *) legendre_psi_dev, 
                      (cuDoubleComplex *) legendre_psi_dev, CUFFT_INVERSE) == CUFFT_SUCCESS);
  
  if(do_scale) {
    const int &n1 = r1.n;
    const int &n2 = r2.n;
    const int n_legs = l_max + 1;
    
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(cublas_handle, n1*n2*n_legs, &s, (cuDoubleComplex *) legendre_psi_dev, 1) 
           == CUBLAS_STATUS_SUCCESS);
  }
}

void OmegaWavepacket::copy_psi_from_device_to_host()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  insist(psi && psi_dev);
  checkCudaErrors(cudaMemcpyAsync(psi, psi_dev, n1*n2*n_theta*sizeof(Complex), cudaMemcpyDeviceToHost));
}

void OmegaWavepacket::copy_psi_from_host_to_device()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  if(!psi_dev) 
    checkCudaErrors(cudaMalloc(&psi_dev, n1*n2*n_theta*sizeof(Complex)));
  insist(psi_dev);
  
  checkCudaErrors(cudaMemcpyAsync(psi_dev, psi, n1*n2*n_theta*sizeof(Complex), cudaMemcpyHostToDevice));
}


void OmegaWavepacket::calculate_wavepacket_module_for_legendre_psi()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;
  
  const Complex *legendre_psi_dev_ = legendre_psi_dev + omega*n1*n2;
  
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
  const int n = n1*n2*n_theta;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
  
  _evolution_with_potential_<<<n_blocks, n_threads>>>(psi_dev, pot_dev, n, dt);
}

void OmegaWavepacket::evolution_with_kinetic(const double dt)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;
  
  const int n = n1*n2*n_legs;

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
  
  const int n = n1*n2*n_legs;

  Complex *legendre_psi_dev_ = legendre_psi_dev + n1*n2*omega;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
  
  _evolution_with_rotational_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
    (legendre_psi_dev_, n1, n2, n_legs, dt);
}

void OmegaWavepacket::calculate_kinetic_energy_for_legendre_psi()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int l = 0; l < n_legs; l++) {
    
    const cuDoubleComplex *legendre_psi_dev_ = (cuDoubleComplex *) legendre_psi_dev + n1*n2*(l+omega);
    
    _psi_times_kinitic_energy_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
      ((Complex *) psi_tmp_dev, (const Complex *) legendre_psi_dev_, n1, n2);
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, legendre_psi_dev_, 1, psi_tmp_dev, 1, 
                       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    sum += dot.real();
  }
  
  sum *= r1.dr*r2.dr/n1/n2;
  
  _kinetic_energy = sum;
}

void OmegaWavepacket::calculate_rotational_energy_for_legendre_psi()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;

  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int l = 0; l < n_legs; l++) {
    const cuDoubleComplex *legendre_psi_in_dev = (cuDoubleComplex *) legendre_psi_dev + n1*n2*(l+omega);
    
    _psi_times_moments_of_inertia_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
      ((Complex *) psi_tmp_dev, (const Complex *) legendre_psi_in_dev, n1, n2);
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, legendre_psi_in_dev, 1, psi_tmp_dev, 1, 
                       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    sum += (l+omega)*(l+omega+1)*dot.real();
  }
  
  sum *= r1.dr*r2.dr;
  
  _rotational_energy = sum;
}

void OmegaWavepacket::evolution_with_coriolis(const double dt, 
					      const int l, const int omega1,
					      const double *coriolis_matrices_dev,
					      const Complex *legendre_psi_omega1,
					      cudaStream_t *stream)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;

  Complex *legendre_psi_dev_ = psi_dev + l*n1*n2;;
  
  const int n = coriolis_matrices[l].omega_max - coriolis_matrices[l].omega_min + 1;
  const double *e = coriolis_matrices_dev + coriolis_matrices[l].offset;
  const double *v = e + n;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);

  if(stream) 
    _evolution_with_coriolis_<<<n_blocks, n_threads, n1*sizeof(Complex), *stream>>>
      (legendre_psi_dev_, n1, n2, e, v, n, omega, omega1, dt, legendre_psi_omega1);
  else
    _evolution_with_coriolis_<<<n_blocks, n_threads, n1*sizeof(Complex)>>>
      (legendre_psi_dev_, n1, n2, e, v, n, omega, omega1, dt, legendre_psi_omega1);
  
}

void OmegaWavepacket::evolution_with_coriolis(const double dt, const int omega1,
					      const double *coriolis_matrices_dev,
					      const Complex *legendre_psi_omega1)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;

  const int n_streams = l_max + 1;
  
  cudaStream_t *streams = (cudaStream_t *) malloc(n_streams*sizeof(cudaStream_t));
  insist(streams);
  for(int i = 0; i < n_streams; i++) 
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  
  int i_stream = 0;
  for(int l = 0; l <= l_max; l++) {
    if(coriolis_matrices[l].l == -1) continue;
    
    evolution_with_coriolis(dt, l, omega1, coriolis_matrices_dev,
			    legendre_psi_omega1+l*n1*n2, &streams[i_stream]);
    i_stream++;
  }
  insist(i_stream <= n_streams);
  
  checkCudaErrors(cudaDeviceSynchronize());
  
  for(int i = 0; i < n_streams; i++) 
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  if(streams) { free(streams); streams = 0; }
}

void OmegaWavepacket::zero_psi_dev()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int n = n1*n2*n_theta;
  checkCudaErrors(cudaMemset(psi_dev, 0, n*sizeof(Complex)));
}

void OmegaWavepacket::update_evolution_with_coriolis()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max - omega + 1;
  
  checkCudaErrors(cudaMemcpy(legendre_psi_dev + n1*n2*omega, psi_dev + n1*n2*omega, 
			     n1*n2*n_legs*sizeof(Complex), cudaMemcpyDeviceToDevice));
}

void OmegaWavepacket::calculate_coriolis_energy_for_legendre_psi(const int omega1,
								 const double *coriolis_matrices_dev,
								 const Complex *legendre_psi_omega1)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  
  Complex *C = 0;
  checkCudaErrors(cudaMalloc(&C, n1*(l_max+1)*sizeof(Complex)));
  checkCudaErrors(cudaMemset(C, 0, n1*(l_max+1)*sizeof(Complex)));
  insist(C);
  
  Complex *psi_dot_tmp = 0;
  checkCudaErrors(cudaMalloc(&psi_dot_tmp, n1*(l_max+1)*sizeof(Complex)));
  checkCudaErrors(cudaMemset(psi_dot_tmp, 0, n1*(l_max+1)*sizeof(Complex)));

  for(int l = 0; l <= l_max; l++) {
    
    if(coriolis_matrices[l].l == -1) continue;
    
    const int n = coriolis_matrices[l].omega_max - coriolis_matrices[l].omega_min + 1;
    const double *e = coriolis_matrices_dev + coriolis_matrices[l].offset;
    const double *v = e + n;
    
    const int n_threads = _NTHREADS_;
    const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1);  
    _calculate_coriolis_on_device_<<<n_blocks, n_threads>>>(n, e, v, omega, omega1, C + l*n1);
  }
  
  insist(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS);

  for(int l = 0; l <= l_max; l++) {
    
    if(coriolis_matrices[l].l == -1) continue;
    
    for(int i = 0; i < n1; i++) {
      
      const Complex *legendre_psi_dev_ = legendre_psi_dev + l*n1*n2 + i;
      const Complex *legendre_psi_omega1_ = legendre_psi_omega1 + l*n1*n2 + i;
      
      const int k = cudaMath::ij_2_index(n1, l_max+1, i, l);

      insist(cublasZdotc(cublas_handle, n2, 
			 (const cuDoubleComplex *) legendre_psi_dev_, n1, 
			 (const cuDoubleComplex *) legendre_psi_omega1_, n1, 
			 (cuDoubleComplex *) psi_dot_tmp+k ) == CUBLAS_STATUS_SUCCESS);
    }
  }

  checkCudaErrors(cudaDeviceSynchronize());
  
  insist(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST) == CUBLAS_STATUS_SUCCESS);
  
  insist(cublasSetStream(cublas_handle, NULL) == CUBLAS_STATUS_SUCCESS);

  Complex dot(0.0, 0.0);
  
  insist(cublasZdotu(cublas_handle, n1*(l_max+1), 
		     (const cuDoubleComplex *) C, 1, 
		     (const cuDoubleComplex *) psi_dot_tmp, 1, 
		     (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);

  _coriolis_energy += dot.real()*r1.dr*r2.dr;
    
  _CUDA_FREE_(C);
  _CUDA_FREE_(psi_dot_tmp);
}


void OmegaWavepacket::calculate_coriolis_energy_for_legendre_psi_2(const int omega1,
								   const double *coriolis_matrices_dev,
								   const Complex *legendre_psi_omega1)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_legs = l_max + 1;

  double *C = 0;
  checkCudaErrors(cudaMalloc(&C, n1*(l_max+1)*sizeof(double)));
  checkCudaErrors(cudaMemset(C, 0, n1*(l_max+1)*sizeof(double)));
  insist(C);
  
  for(int l = 0; l <= l_max; l++) {
    
    if(coriolis_matrices[l].l == -1) continue;
    
    const int n = coriolis_matrices[l].omega_max - coriolis_matrices[l].omega_min + 1;
    const double *e = coriolis_matrices_dev + coriolis_matrices[l].offset;
    const double *v = e + n;
    
    const int n_threads = _NTHREADS_;
    const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1);  
    _calculate_coriolis_on_device_<<<n_blocks, n_threads>>>(n, e, v, omega, omega1, C + l*n1);
  }

  Complex *tmp = 0;
  checkCudaErrors(cudaMalloc(&tmp, n1*n2*(l_max+1)*sizeof(Complex)));
  insist(tmp);
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2*n_legs); 
  
  _coriolis_matrices_production_<<<n_blocks, n_threads>>>(legendre_psi_omega1, C, tmp, n1, n2, n_legs);
  
  Complex dot(0.0, 0.0);
  
  insist(cublasZdotc(cublas_handle, n1*n2*n_legs, 
		     (const cuDoubleComplex *) legendre_psi_dev, 1, 
		     (const cuDoubleComplex *) tmp, 1, 
		     (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
  
  if(omega != omega1)
    dot *= 2;
  
  _coriolis_energy += dot.real()*r1.dr*r2.dr;
  
  _CUDA_FREE_(C);
  _CUDA_FREE_(tmp);
}
