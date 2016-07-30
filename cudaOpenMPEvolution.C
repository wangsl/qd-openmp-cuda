

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

void omegas_test(OmegaStates &omegas);

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

  CudaOpenMPQMMD evolCUDA(pot.data, r1, r2, theta, omegas, time);
  evolCUDA.test();
 
  std::cout.flush();
  std::cout.precision(np);
}
