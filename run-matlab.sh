#!/bin/bash

module purge
module load intel/16.0.3
module load gcc/4.8.2
module load cuda/7.5.18
module load matlab/2015b

export LD_PRELOAD=$GCC_LIB/libstdc++.so:$LD_PRELOAD

#taskset -c 0-9 
matlab -nodisplay -r "FH2main; exit"

