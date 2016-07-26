

/* $Id$ */

#include <iostream>
#include <cstring>
#include <cmath>
#include <mex.h>

#include "fort.h"
#include "matlabUtils.h"
#include "matlabStructures.h"
#include "matlabArray.h"

#include "cudaOpenMP.h"

#include "evolutionUtils.h"

void omegas_test_3(OmegaStates &omegas);
void omegas_test_4(OmegaStates &omegas);
void omegas_test_5(OmegaStates &omegas);

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  const int np = std::cout.precision();
  std::cout.precision(12);
  
  std::cout << " Quantum Dynamics Time evolotion with CUDA and OpenMP" << std::endl;

  insist(nrhs == 1);

  mxArray *mxPtr = 0;
  
  mxPtr = mxGetField(prhs[0], 0, "r1");
  insist(mxPtr);
  RadialCoordinate r1(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "r2");
  insist(mxPtr);
  RadialCoordinate r2(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "theta");
  insist(mxPtr);
  AngleCoordinate theta(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "pot");
  insist(mxPtr);
  MatlabArray<double> pot(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "psi");
  insist(mxPtr);
  MatlabArray<Complex> psi(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "time");
  insist(mxPtr);
  EvolutionTime time(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "options");
  insist(mxPtr);
  Options options(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "dump1");
  insist(mxPtr);
  DumpFunction dump1(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "dump2");
  insist(mxPtr);
  DumpFunction dump2(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "CRP");
  insist(mxPtr);
  CummulativeReactionProbabilities CRP(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "OmegaStates");
  insist(mxPtr);
  OmegaStates omegas(mxPtr);

  std::cout << omegas << std::endl;

  omegas_test_4(omegas);
  omegas_test_5(omegas);

#if 0
  for(int j = omegas.omegas[0]; j < omegas.l_max; j++) {
    RMat cor_mat;
    setup_coriolis_matrix(omegas.J, omegas.parity, j, cor_mat);
    std::cout << cor_mat << std::endl;
  }
#endif
  
#if 0
  CudaOpenMPQMMD evolCUDA(pot.data, r1, r2, theta, omegas, time);
  evolCUDA.test();
#endif
 
  std::cout.flush();
  std::cout.precision(np);
}
