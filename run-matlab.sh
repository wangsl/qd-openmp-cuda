#!/bin/bash

module purge
module load intel/16.0.3
module load gcc/4.8.2
module load cuda/7.5.18
module load matlab/2015b

#export LD_PRELOAD=$LD_PRELOAD:$MKL_LIB/libmkl_intel_ilp64.so:$MKL_LIB/libmkl_core.so:$MKL_LIB/libmkl_intel_thread.so:$INTEL_LIB/libiomp5.so

export LD_PRELOAD=$GCC_LIB/libstdc++.so:$LD_PRELOAD

#export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="1"

taskset -c 0-19 matlab -nodisplay -r "FH2main; exit" 2>&1 | tee stdout.log
