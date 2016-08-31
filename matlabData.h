
#ifndef MATLAB_DATA_H
#define MATLAB_DATA_H

#include "evolutionUtils.h"

namespace MatlabData
{
  const double *dump1();
  void dump1(const double *d);

  const double *dump2();
  void dump2(const double *d);

  inline int dump_wavepacket() { return dump1() && dump2() ? 1 : 0; }
};

#endif /* MATLAB_DATA_H */
