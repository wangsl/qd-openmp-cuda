
/* $Id$ */

#include "matlabStructures.h"
#include "matlabUtils.h"
#include "matlabArray.h"
//#include "fftwinterface.h"

void remove_matlab_script_extension(char *script, const char *extension)
{
  insist(script);
  insist(extension);
  const int len = strlen(script) - strlen(extension);
  if(!strcmp((const char *) script+len, extension)) 
    ((char *) script)[len] = '\0';
}

RadialCoordinate::RadialCoordinate(const mxArray *mx) :
  mx(mx),
  n(*(int *) mxGetData(mx, "n")),
  left(*(double *) mxGetData(mx, "left")),
  dr(*(double *) mxGetData(mx, "dr")),
  mass(*(double *) mxGetData(mx, "mass")),
  dump_Cd(*(double *) mxGetData(mx, "dump_Cd")),
  dump_xd(*(double *) mxGetData(mx, "dump_xd"))
{ }

AngleCoordinate::AngleCoordinate(const mxArray *mx) :
  mx(mx),
  n(*(int *) mxGetData(mx, "n")),
  m(*(int *) mxGetData(mx, "m"))
{
  x = RVec(n, (double *) mxGetData(mx, "x"));
  w = RVec(n, (double *) mxGetData(mx, "w"));
  
  double *p = (double *) mxGetData(mx, "associated_legendre");
  insist(p);

  associated_legendre = RMat(m+1, n, p);
}
  
EvolutionTime::EvolutionTime(const mxArray *mx) :
  mx(mx),
  total_steps(*(int *) mxGetData(mx, "total_steps")),
  steps(*(int *) mxGetData(mx, "steps")),
  time_step(*(double *) mxGetData(mx, "time_step"))
{ }

Options::Options(const mxArray *mx) :
  mx(mx),
  wave_to_matlab(0),
  test_name(0),
  steps_to_copy_psi_from_device_to_host(*(int *) mxGetData(mx, "steps_to_copy_psi_from_device_to_host"))
{
  wave_to_matlab = mxGetString(mx, "wave_to_matlab");
  if(wave_to_matlab)
    remove_matlab_script_extension(wave_to_matlab, ".m");
  
  test_name = mxGetString(mx, "test_name");
}

Options::~Options()
{
  if(wave_to_matlab) { delete [] wave_to_matlab; wave_to_matlab = 0; }
  if(test_name) { delete [] test_name; test_name = 0; }
}

DumpFunction::DumpFunction(const mxArray *mx) :
  mx(mx), dump(0)
{
  dump = (double *) mxGetData(mx, "dump");
}

DumpFunction::~DumpFunction()
{
  if(dump) dump = 0;
}

CummulativeReactionProbabilities::CummulativeReactionProbabilities(const mxArray *mx) :
  mx(mx),
  n_dividing_surface(*(int *) mxGetData(mx, "n_dividing_surface")),
  n_gradient_points(*(int *) mxGetData(mx, "n_gradient_points")),
  n_energies(*(int *) mxGetData(mx, "n_energies")),
  calculate_CRP(*(int *) mxGetData(mx, "calculate_CRP"))
{
  energies = RVec(n_energies, (double *) mxGetData(mx, "energies"));
  eta_sq = RVec(n_energies, (double *) mxGetData(mx, "eta_sq"));
  CRP = RVec(n_energies, (double *) mxGetData(mx, "CRP"));
}

OmegaStates::OmegaStates(const mxArray *mx) :
  mx(mx), 
  J(*(int *) mxGetData(mx, "J")),
  parity(*(int *) mxGetData(mx, "parity")),
  l_max(*(int *) mxGetData(mx, "l_max")),
  n_omegas_max(*(int *) mxGetData(mx, "n_omegas_max"))
{
  int n1 = 0;
  int n2 = 0;
  int n3 = 0;

  const mxArray *omegas_ptr = mxGetField(mx, 0, "omegas");
  insist(omegas_ptr);

  const MatlabArray<int> omegas_(omegas_ptr);
  const size_t *dims_ = omegas_.dims();
  omegas = Vec<int>(dims_[0]*dims_[1], omegas_.data);

  insist(n_omegas_max >= omegas.size());

  const mxArray *assL_ptr = mxGetField(mx, 0, "associated_legendres");
  insist(assL_ptr);
  
  const MatlabArray<double> ass_leg(assL_ptr);
  dims_ = ass_leg.dims();
  
  insist(ass_leg.n_dims() == 2 || ass_leg.n_dims() == 3);
  
  n1 = dims_[0];
  n2 = dims_[1];
  n3 = ass_leg.n_dims() == 3 ? dims_[2] : 1;

  associated_legendres.resize(n3);
  
  const double *p = ass_leg.data;
  for(int i = 0; i < n3; i++) {
    associated_legendres[i] = RMat(n1, n2-i, const_cast<double *>(p));
    p += n1*n2;
  }
  
  const mxArray *wp_ptr = mxGetField(mx, 0, "wave_packets");
  insist(wp_ptr);
  
  const MatlabArray<Complex> wp(wp_ptr);
  dims_ = wp.dims();
  
  n1 = dims_[0];
  n2 = dims_[1];
  n3 = ass_leg.n_dims() == 3 ? dims_[2] : 1;
  
  wave_packets.resize(n3);
  
  const size_t size = n1*n2*n3/2;
  insist(2*size ==  n1*n2*n3);
  const Complex *c_p = wp.data;
  for(int i = 0; i < n3; i++) {
    wave_packets[i] = Vec<Complex>(size, const_cast<Complex *>(c_p));
    c_p += size;
  }
}  

OmegaStates::~OmegaStates()
{  if(mx) mx = 0; }

