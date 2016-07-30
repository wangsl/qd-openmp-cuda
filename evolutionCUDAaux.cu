
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

