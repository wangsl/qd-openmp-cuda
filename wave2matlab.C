
/* $Id$ */

#include <iostream>
#include <mex.h>

#include <unistd.h>

#include "str.h"
#include "matlabUtils.h"

int file_exist(const char *file_name)
{
  return access(file_name, F_OK) ? 0 : 1;
}

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
