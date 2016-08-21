
#include "cudaMath.h"
#include "evolutionUtils.h"

static __global__ void _evolution_with_potential_(Complex *psi, const double *pot, 
						  const int n, const double dt)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < n) 
    psi[index] *= exp(Complex(0.0, -dt)*pot[index]);
}

static __global__ void _evolution_with_kinetic_(Complex *psi, const int n1, const int n2, const int m, 
                                                const double dt)
{
  extern __shared__ double s_data[];
  
  double *kin1 = (double *) s_data;
  double *kin2 = (double *) &kin1[n1];
  
  cudaMath::setup_kinetic_energy_for_fft(kin1, r1_dev.n, r1_dev.n*r1_dev.dr, r1_dev.mass);
  cudaMath::setup_kinetic_energy_for_fft(kin2, r2_dev.n, r2_dev.n*r2_dev.dr, r2_dev.mass);
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*m) {
    int i = -1; int j = -1; int k = -1;
    cudaMath::index_2_ijk(index, n1, n2, m, i, j, k);
    psi[index] *= exp(Complex(0.0, -dt)*(kin1[i]+kin2[j]));
  }
}

static __global__ void _evolution_with_rotational_(Complex *psi, const int n1, const int n2, const int m,
                                                   const double dt)
{
  extern __shared__ double s_data[];
  
  double *I1 = (double *) s_data;
  double *I2 = (double *) &I1[n1];
  
  cudaMath::setup_moments_of_inertia(I1, r1_dev.n, r1_dev.left, r1_dev.dr, r1_dev.mass);
  cudaMath::setup_moments_of_inertia(I2, r2_dev.n, r2_dev.left, r2_dev.dr, r2_dev.mass);
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*m) {
    int i = -1; int j = -1; int l = -1;
    cudaMath::index_2_ijk(index, n1, n2, m, i, j, l);
    psi[index] *= exp(-Complex(0.0, 1.0)*dt*l*(l+1)*(I1[i]+I2[j]));
  }
}

static __global__ void _psi_times_kinitic_energy_(Complex *psiOut, const Complex *psiIn, 
                                                  const int n1, const int n2)
{
  extern __shared__ double s_data[];
  
  double *kin1 = (double *) s_data;
  double *kin2 = (double *) &kin1[n1];
  
  cudaMath::setup_kinetic_energy_for_fft(kin1, r1_dev.n, r1_dev.n*r1_dev.dr, r1_dev.mass);
  cudaMath::setup_kinetic_energy_for_fft(kin2, r2_dev.n, r2_dev.n*r2_dev.dr, r2_dev.mass);
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2) {
    int i = -1; int j = -1;
    cudaMath::index_2_ij(index, n1, n2, i, j);
    psiOut[index] = psiIn[index]*(kin1[i] + kin2[j]);
  }
}

static __global__ void _psi_times_moments_of_inertia_(Complex *psiOut, const Complex *psiIn, 
						      const int n1, const int n2)
{
  extern __shared__ double s_data[];
  
  double *I1 = (double *) s_data;
  double *I2 = (double *) &I1[n1];
  
  cudaMath::setup_moments_of_inertia(I1, r1_dev.n, r1_dev.left, r1_dev.dr, r1_dev.mass);
  cudaMath::setup_moments_of_inertia(I2, r2_dev.n, r2_dev.left, r2_dev.dr, r2_dev.mass);
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2) {
    int i = -1; int j = -1;
    cudaMath::index_2_ij(index, n1, n2, i, j);
    psiOut[index] = psiIn[index]*(I1[i] + I2[j]);
  }
}

static __global__ void _print_coriolis_on_device_(const int n, const double *e, const double *v)
{
  if(e) {
    for(int i = 0; i < n; i++)
      printf("%18.12f", e[i]);
    printf("\n");
  }
  
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) { 
      const int k = cudaMath::ij_2_index(n, n, i, j);
      printf("%18.12f", v[k]);
    }
    printf("\n");
  }
}

static __global__ void _calculate_coriolis_on_device_(const int n, const double *e, 
						      const double *v, double *b)
{
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      const int ij = cudaMath::ij_2_index(n, n, i, j);
      b[ij] = 0;
      for(int alpha = 0; alpha < n; alpha++) {
	const int i_alpha = cudaMath::ij_2_index(n, n, i, alpha);
	const int j_alpha = cudaMath::ij_2_index(n, n, j, alpha);
	b[ij] += e[alpha]*v[i_alpha]*v[j_alpha];
      }
    }
  }
}

static __global__ void _calculate_coriolis_on_device_(const int n, 
						      const double *e, const double *v, 
						      const int omega, const int omega1, 
						      double *b)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < r1_dev.n) {
    
    const double R = r1_dev.left + index*r1_dev.dr;
    const double I = 2*r1_dev.mass*R*R;
    
    const int &i = omega;
    const int &j = omega1;
    
    b[index] = 0.0;
    for(int alpha = 0; alpha < n; alpha++) {
      const int i_alpha = cudaMath::ij_2_index(n, n, i, alpha);
      const int j_alpha = cudaMath::ij_2_index(n, n, j, alpha);
      b[index] += -e[alpha]/I*v[i_alpha]*v[j_alpha];
    }
  }
}

static __global__ void _calculate_coriolis_on_device_(const int n, 
						      const double *e, const double *v, 
						      const int omega, const int omega1, 
						      Complex *b)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < r1_dev.n) {
    
    const double R = r1_dev.left + index*r1_dev.dr;
    const double I = 2*r1_dev.mass*R*R;
    
    const int &i = omega;
    const int &j = omega1;
    
    b[index].zero();
    for(int alpha = 0; alpha < n; alpha++) {
      const int i_alpha = cudaMath::ij_2_index(n, n, i, alpha);
      const int j_alpha = cudaMath::ij_2_index(n, n, j, alpha);
      b[index] += Complex(-e[alpha]/I*v[i_alpha]*v[j_alpha], 0.0);
    }
  }
}

static __global__ void _evolution_with_coriolis_(Complex *psi, const int n1, const int n2,
						const double *e, const double *v, const int n, 
						const int omega, const int omega1,
						const double dt, const Complex *psi1)
{
  extern __shared__ Complex b[];

  for(int i = threadIdx.x; i < n1; i += blockDim.x) {
    
    const double R = r1_dev.left + i*r1_dev.dr;
    const double I = 2*r1_dev.mass*R*R;
    
    const Complex dt_I(0.0, dt/I);
    
    b[i].zero();
    for(int alpha = 0; alpha < n; alpha++) {
      const int omega_alpha = cudaMath::ij_2_index(n, n, omega, alpha);
      const int omega1_alpha = cudaMath::ij_2_index(n, n, omega1, alpha);
      b[i] += exp(dt_I*e[alpha])*v[omega_alpha]*v[omega1_alpha];
    }
  }

  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n1*n2) {
    int i = -1; int j = -1;
    cudaMath::index_2_ij(index, n1, n2, i, j);
    psi[index] += b[i]*psi1[index];
  }
}

static __global__ void _coriolis_matrices_production_(const Complex *p_in, const double *C,
						      Complex *p_out, const int n1, const int n2, const int n3)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n1*n2*n3) {
    int i = -1; int j = -1; int k = -1;
    cudaMath::index_2_ijk(index, n1, n2, n3, i, j, k);
    const int ik = cudaMath::ij_2_index(n1, n3, i, k);
    p_out[index] = p_in[index]*C[ik];
  }
}
