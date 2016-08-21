
/* created at: 2016-08-03 17:02:55 */

#include <iostream>
using namespace std;
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "indent.h"
#include "matlabStructures.h"
#include "die.h"

ostream & operator <<(ostream &s, const RadialCoordinate &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void RadialCoordinate::write_fields(ostream &s) const
{
  s << Indent() << "n " << n << "\n";
  s << Indent() << "left " << left << "\n";
  s << Indent() << "dr " << dr << "\n";
  s << Indent() << "mass " << mass << "\n";
  s << Indent() << "dump_Cd " << dump_Cd << "\n";
  s << Indent() << "dump_xd " << dump_xd << "\n";
}

ostream & operator <<(ostream &s, const AngleCoordinate &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void AngleCoordinate::write_fields(ostream &s) const
{
  s << Indent() << "n " << n << "\n";
  s << Indent() << "m " << m << "\n";
}

ostream & operator <<(ostream &s, const EvolutionTime &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void EvolutionTime::write_fields(ostream &s) const
{
  s << Indent() << "total_steps " << total_steps << "\n";
  s << Indent() << "time_step " << time_step << "\n";
  s << Indent() << "steps " << steps << "\n";
}

ostream & operator <<(ostream &s, const Options &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void Options::write_fields(ostream &s) const
{
  if (wave_to_matlab)
    s << Indent() << "wave_to_matlab " << wave_to_matlab << "\n";
  if (test_name)
    s << Indent() << "test_name " << test_name << "\n";
  s << Indent() << "steps_to_copy_psi_from_device_to_host " << steps_to_copy_psi_from_device_to_host << "\n";
}

ostream & operator <<(ostream &s, const CummulativeReactionProbabilities &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void CummulativeReactionProbabilities::write_fields(ostream &s) const
{
  s << Indent() << "energies " << energies << "\n";
  s << Indent() << "eta_sq " << eta_sq << "\n";
  s << Indent() << "CRP " << CRP << "\n";
  s << Indent() << "n_dividing_surface " << n_dividing_surface << "\n";
  s << Indent() << "n_gradient_points " << n_gradient_points << "\n";
  s << Indent() << "n_energies " << n_energies << "\n";
  s << Indent() << "calculate_CRP " << calculate_CRP << "\n";
}

ostream & operator <<(ostream &s, const OmegaStates &c)
{
  s << " {\n";
  IndentPush();
  c.write_fields(s);
  IndentPop();
  return s << Indent() << " }";
}

void OmegaStates::write_fields(ostream &s) const
{
  s << Indent() << "J " << J << "\n";
  s << Indent() << "parity " << parity << "\n";
  s << Indent() << "l_max " << l_max << "\n";
  s << Indent() << "n_omegas_max " << n_omegas_max << "\n";
  s << Indent() << "omegas " << omegas << "\n";
}

