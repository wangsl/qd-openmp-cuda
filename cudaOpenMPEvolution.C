
/* $Id$ */

#include <iostream>
#include <cstring>
#include <cmath>
#include <mex.h>

#include "matlabUtils.h"
#include "matlabStructures.h"
#include "matlabArray.h"
#include "matlabData.h"
#include "cudaOpenMP.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  const int np = std::cout.precision();
  std::cout.precision(12);
  
  std::cout << " Quantum Dynamics Time evolotion with CUDA and OpenMP\n" << std::endl;

  insist(nrhs == 1);

  mxArray *mxPtr = 0;
  
  mxPtr = mxGetField(prhs[0], 0, "r1");
  insist(mxPtr);
  MatlabData::r1(new RadialCoordinate(mxPtr));
  
  mxPtr = mxGetField(prhs[0], 0, "r2");
  insist(mxPtr);
  MatlabData::r2(new RadialCoordinate(mxPtr));
  
  mxPtr = mxGetField(prhs[0], 0, "theta");
  insist(mxPtr);
  MatlabData::theta(new AngleCoordinate(mxPtr));
  
  mxPtr = mxGetField(prhs[0], 0, "pot");
  insist(mxPtr);
  MatlabArray<double> pot(mxPtr);
  MatlabData::potential(pot.data);
  
  mxPtr = mxGetField(prhs[0], 0, "time");
  insist(mxPtr);
  EvolutionTime time(mxPtr);
  MatlabData::time(new EvolutionTime(mxPtr));
  
  mxPtr = mxGetField(prhs[0], 0, "options");
  insist(mxPtr);
  Options options(mxPtr);
  MatlabData::options(new Options(mxPtr));
  
  mxPtr = mxGetField(prhs[0], 0, "dump1");
  insist(mxPtr);
  DumpFunction dump1(mxPtr);
  MatlabData::dump1(dump1.dump);
  
  mxPtr = mxGetField(prhs[0], 0, "dump2");
  insist(mxPtr);
  DumpFunction dump2(mxPtr);
  MatlabData::dump2(dump2.dump);
  
  mxPtr = mxGetField(prhs[0], 0, "CRP");
  insist(mxPtr);
  CummulativeReactionProbabilities CRP(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "OmegaStates");
  insist(mxPtr);
  MatlabData::omega_states(new OmegaStates(mxPtr));
  
  CudaOpenMPQMMD evolCUDA;
  //evolCUDA.test();
  evolCUDA.test_multiple_cards();
  //evolCUDA.p2p_test();
 
  std::cout.flush();
  std::cout.precision(np);
}
