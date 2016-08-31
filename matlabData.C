

#include "matlabData.h"

static const double *_dump1 = 0;
static const double *_dump2 = 0;

// dump1
const double * MatlabData::dump1() { return _dump1; }

void MatlabData::dump1(const double *d) 
{
  insist(!_dump1);
  _dump1 = d;
}

// dump2
const double * MatlabData::dump2() { return _dump2; }

void MatlabData::dump2(const double *d) 
{
  insist(!_dump2);
  _dump2 = d;
}

