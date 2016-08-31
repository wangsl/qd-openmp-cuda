
/* $Id$ */

#include <mex.h>
#include "matlabUtils.h"
#include "fort.h"

void MatlabCrashLoc(const char *message, const char *file_name, const int line)
{
  char buf[256];
  sprintf(buf, "Matlab error in module %s, line %d\n %s", file_name, line, message);
  mexErrMsgTxt(buf);
}

// Fortran version
extern "C" void FORT(matlabcrashloc)(const char *message, const char *file_name, const int &line)
{
  MatlabCrashLoc(message, file_name, line);
}
