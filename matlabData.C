
#include "matlabData.h"

static const RadialCoordinate *_r1 = 0;
static const RadialCoordinate *_r2 = 0;
static const AngleCoordinate *_theta = 0;

static const double *_potential = 0;

static const double *_dump1 = 0;
static const double *_dump2 = 0;

static EvolutionTime *_time = 0;

static OmegaStates *_omega_states = 0;

static const Options *_options = 0;

// r1
const RadialCoordinate *MatlabData::r1() { return _r1; }
void MatlabData::r1(const RadialCoordinate *r) { insist(r); _r1 = r; }

// r2
const RadialCoordinate *MatlabData::r2() { return _r2; }
void MatlabData::r2(const RadialCoordinate *r) { insist(r); _r2 = r; }

// theta
const AngleCoordinate *MatlabData::theta() { return _theta; }
void MatlabData::theta(const AngleCoordinate *th) { insist(th); _theta = th; }

// potential
const double *MatlabData::potential() { return _potential; }
void MatlabData::potential(const double *p) { insist(!_potential); _potential = p; }

// dump1
const double *MatlabData::dump1() { return _dump1; }
void MatlabData::dump1(const double *d) { insist(!_dump1); _dump1 = d; }

// dump2
const double *MatlabData::dump2() { return _dump2; }
void MatlabData::dump2(const double *d) { insist(!_dump2); _dump2 = d; }

// evolution time
EvolutionTime *MatlabData::time() { return _time; }
void MatlabData::time(EvolutionTime *t) { insist(!_time); _time = t; }

// Options
const Options *MatlabData::options() { return _options; }
void MatlabData::options(const Options *o) { insist(!_options); _options = o; }

// Omega stats
OmegaStates *MatlabData::omega_states() { return _omega_states; }
void MatlabData::omega_states(OmegaStates *om) { insist(!_omega_states);  _omega_states = om; }
