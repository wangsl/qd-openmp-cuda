
#ifndef EVOLUTION_UTILS_H
#define EVOLUTION_UTILS_H

#include "matlabStructures.h"

struct CoriolisMatrixAux
{
  int l = -1;
  int omega_min = -1;
  int omega_max = -1;
  int offset = -1;
};

inline std::ostream & operator <<(std::ostream &s, const CoriolisMatrixAux &c)
{
  return s << " l: " << c.l << " omega_min: " <<  c.omega_min << " omega_max: " << c.omega_max
	   << " offset: " << c.offset;
}

void calculate_coriolis_matrix_dimension(const int J, const int p, const int j, int &omega_min, int &omega_max);
void setup_coriolis_matrix(const int J, const int p, const int j, RMat &cor_mat);

#ifdef __NVCC__

namespace EvolutionUtils {
  
  struct RadialCoordinate
  {
    double left;
    double dr;
    double mass;
    double dump_Cd;
    double dump_xd;
    int n;
  };
  
  inline void copy_radial_coordinate_to_device(const RadialCoordinate &r_dev, 
                                               const double &left, const double &dr,
                                               const double &mass, 
                                               const double &dump_Cd, const double &dump_xd,
                                               const int &n)
  {
    RadialCoordinate r;
    r.left = left;
    r.dr = dr;
    r.mass = mass;
    r.dump_Cd = dump_Cd;
    r.dump_xd = dump_xd;
    r.n = n;
    checkCudaErrors(cudaMemcpyToSymbol(r_dev, &r, sizeof(RadialCoordinate)));
  }
}

// device constant memory variables are defined as wavepackets1device.cu

extern __constant__ EvolutionUtils::RadialCoordinate r1_dev;
extern __constant__ EvolutionUtils::RadialCoordinate r2_dev;

extern __constant__ double dump1_dev[1024];
extern __constant__ double dump2_dev[1024];

#endif /* __NVCC__ */

#endif /* EVOLUTION_UTILS_H */
