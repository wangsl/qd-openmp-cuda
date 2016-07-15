
static __global__ void _evolution_with_potential_(Complex *psi, const double *pot, 
						  const int n, const double dt)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < n) 
    psi[index] *= exp(Complex(0.0, -dt)*pot[index]);
}

