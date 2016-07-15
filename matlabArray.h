
/* $Id$ */

#ifndef MATLAB_ARRAY_H
#define MATLAB_ARRAY_H

#include <iostream>
#include <mex.h>
#include "matlabUtils.h"

template<class T> class MatlabArray
{
public:
  const mxArray *mx;
  T *data;
  
  MatlabArray(const mxArray *mx_) :
    mx(mx_), data(0)
  {    
    data = (T *) mxGetData(mx);
    insist(data);
  }
  
  int n_dims() const
  { return mx ? mxGetNumberOfDimensions(mx) : 0; }
  
  const size_t *dims() const
  { return mx ? mxGetDimensions(mx) : 0; }
  
  ~MatlabArray()
  { 
    mx = 0;
    data = 0;
  }
};

#endif /* MATLABARRAY_H */
