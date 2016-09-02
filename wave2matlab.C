
/* $Id$ */

#include <iostream>
#include <mex.h>

#include "str.h"
#include "matlabUtils.h"

void wavepacket_to_matlab(const char *script, const int nrhs, mxArray *prhs[])
{
  if(!file_exist(script + Str(".m"))) return;
  
  std::cout << " Matlab script " << script << std::endl;

  insist(!mexCallMATLAB(0, NULL, nrhs, prhs, script));
}

void wavepacket_to_matlab(const char *script)
{
  if(!file_exist(script + Str(".m"))) return;

  std::cout << " Matlab script " << script << std::endl;

  insist(!mexCallMATLAB(0, NULL, 0, NULL, script));
}
