
/* $Id$ */

#ifndef MATLAB_STRUCTURES_H
#define MATLAB_STRUCTURES_H

#include <iostream>
using namespace std;
#include <mex.h>
#include "rmat.h"
#include "complex.h"
#include "matlabUtils.h"

class RadialCoordinate
{
public:
  
  const int &n; // out
  const double &left; // out
  const double &dr; // out
  const double &mass; // out
  
  RadialCoordinate(const mxArray *mx);
  ~RadialCoordinate() { if(mx) mx = 0; }
  
private:
  const mxArray *mx;
  
  // to prevent assigment and copy operation
  RadialCoordinate(const RadialCoordinate &);
  RadialCoordinate & operator =(const RadialCoordinate &);
  
  /* IO */
  friend ostream & operator <<(ostream &s, const RadialCoordinate &c);
  void write_fields(ostream &s) const;
};

class AngleCoordinate
{
public:
  
  const int &n; // out
  const int &m; // out
  RVec x;
  RVec w;
  RMat associated_legendre; 
  
  AngleCoordinate(const mxArray *mx);
  ~AngleCoordinate() { if(mx) mx = 0; }
  
private:
  const mxArray *mx;

  // to prevent assigment and copy operation
  AngleCoordinate(const AngleCoordinate &);
  AngleCoordinate & operator =(const AngleCoordinate &);
  
  /* IO */
  friend ostream & operator <<(ostream &s, const AngleCoordinate &c);
  void write_fields(ostream &s) const;
};

class EvolutionTime
{
public:

  const int &total_steps; // out
  const double &time_step; // out
  int &steps; // out
  
  EvolutionTime(const mxArray *mx);
  ~EvolutionTime() { if(mx) mx = 0; }
  
private:
  const mxArray *mx;

  EvolutionTime(const EvolutionTime &);
  EvolutionTime & operator =(const EvolutionTime &);
  
  /* IO */
  friend ostream & operator <<(ostream &s, const EvolutionTime &c);
  void write_fields(ostream &s) const;
};

class Options
{
public:
  
  char *wave_to_matlab; // out
  char *test_name; // out
  const int &steps_to_copy_psi_from_device_to_host; // out
  const int &use_p2p_async; // out

  Options(const mxArray *mx);
  ~Options();

private:

  const mxArray *mx;

  Options(const Options &);
  Options & operator =(const Options &);

  friend ostream & operator <<(ostream &s, const Options &c);
  void write_fields(ostream &s) const;
};

class DumpFunction
{
public:
  
  DumpFunction(const mxArray *mx);
  ~DumpFunction();

  const double *dump; 

private:
  const mxArray *mx;
};

class CummulativeReactionProbabilities
{
public:

  RVec energies; // out
  RVec eta_sq; // out
  RVec CRP; // out

  const int &n_dividing_surface; // out
  const int &n_gradient_points; // out
  const int &n_energies; // out
  const int &calculate_CRP; // out
  
  CummulativeReactionProbabilities(const mxArray *mx);

private:
  const mxArray *mx;

  CummulativeReactionProbabilities(const CummulativeReactionProbabilities &);
  CummulativeReactionProbabilities & operator =(const CummulativeReactionProbabilities &);

  friend ostream & operator <<(ostream &s, const CummulativeReactionProbabilities &c);
  void write_fields(ostream &s) const;
  
};

class OmegaStates
{
public:
  
  OmegaStates(const mxArray *mx);
  ~OmegaStates();

  const int &J; // out
  const int &parity; // out
  const int &l_max; // out
  const int &n_omegas_max; // out
  Vec<int> omegas; // out
  Vec<RMat> associated_legendres; 
  Vec<Vec<Complex> > wave_packets;

private:
  const mxArray *mx;

  friend ostream & operator <<(ostream &s, const OmegaStates &c);
  void write_fields(ostream &s) const;
};

#endif /* MATLAB_STRUCTURES_H */
