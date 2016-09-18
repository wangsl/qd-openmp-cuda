
#ifndef MATLAB_DATA_H
#define MATLAB_DATA_H

#include "evolutionUtils.h"

namespace MatlabData
{
  const RadialCoordinate *r1();
  void r1(const RadialCoordinate *r);
  
  const RadialCoordinate *r2();
  void r2(const RadialCoordinate *r);
  
  const AngleCoordinate *theta();
  void theta(const AngleCoordinate *th);
  
  const double *potential();
  void potential(const double *p);

  const double *dump1();
  void dump1(const double *d);

  const double *dump2();
  void dump2(const double *d);

  inline int dump_wavepacket() { return dump1() && dump2() ? 1 : 0; }
  
  EvolutionTime *time();
  void time(EvolutionTime *t);
  
  const Options *options();
  void options(const Options *op);

  OmegaStates *omega_states();
  void omega_states(OmegaStates *om);
};

#endif /* MATLAB_DATA_H */
