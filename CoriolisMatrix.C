
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "rmat.h"
#include "fort.h"

#include "evolutionUtils.h"

/*
  Refercens:
  J. Phys. Chem. A, 102, 9372-9379 (1998)
  John Zhang  Theory and Application of Quantum Molecular Dynamics P343
  http://www.netlib.org/lapack/explore-html/d7/d48/dstev_8f.html
*/

extern "C" void FORT(dstev)(const char *JOBZ, const FInt &N, double *D, 
			    double *E, double *Z, const FInt &LDZ, 
			    double *work, FInt &info);

inline double lambda(const int J, const int Omega, const int sign)
{
  double c = 0.0;
  if(J >= Omega) {
    c = sign%2 == 0 ? 
      sqrt(J*(J+1.0) - Omega*(Omega+1.0)) :
      sqrt(J*(J+1.0) - Omega*(Omega-1.0)) ;
  }
  
  return c;
}

inline int kronecker_delta(const int &a, const int &b)
{
  return a == b ? 1 : 0;
}

int coriolis_matrix_dimension(const int J, const int p, const int j)
{
  if(J == 0) insist(p == 0);
  
  const int omega_min = (J+p)%2 == 0 ? 0 : 1;
  const int omega_max = std::min(J, j);
  
  insist(j >= omega_min);

  const int n = omega_max - omega_min + 1;
  
  return n;
}

void calculate_coriolis_matrix_dimension(const int J, const int p, const int j, 
					 int &omega_min, int &omega_max)
{
  if(J == 0) insist(p == 0);
  
  omega_min = (J+p)%2 == 0 ? 0 : 1;
  omega_max = std::min(J, j);
  
  insist(j >= omega_min);
}

void setup_coriolis_matrix(const int J, const int p, const int j, RMat &cor_mat)
{
  if(J == 0) insist(p == 0);
  
  const int omega_min = (J+p)%2 == 0 ? 0 : 1;
  const int omega_max = std::min(J, j);
  const int n = omega_max - omega_min + 1;

  insist(j >= omega_min);

#if 0
  std::cout << " J: " << J << " p: " << p << " j: " << j
	    << " OmegaMin: " << omega_min << " OmegaMax: " << omega_max 
	    << " size: " << n << std::endl;
#endif
  
  cor_mat.resize(n, n+1);
  
  double *diag_eles = cor_mat;
  for(int k = 0; k < n; k++) {
    const int omega = k + omega_min;
    diag_eles[k] = J*(J+1) - 2*omega*omega;
  }
  
  double *sub_diag_eles = new double [n-1];
  insist(sub_diag_eles);
  for(int k = 0; k < n-1; k++) {
    const int omega = k + omega_min;
    sub_diag_eles[k] = -lambda(J, omega, 0)*lambda(j, omega, 0)*sqrt(1.0 + kronecker_delta(omega, 0));
  }
  
  double *work = new double [std::max(1, 2*(n-1))];
  insist(work);
  
  FInt n_ = n;
  const char jobV[4] = "V";
  FInt info = -100;
  FORT(dstev)(jobV, n_, diag_eles, sub_diag_eles, (double *) cor_mat+n, n_, work, info);
  insist(info == 0);  
  
  if(sub_diag_eles) { delete [] sub_diag_eles; sub_diag_eles = 0; }
  if(work) { delete [] work; work = 0; }
}

