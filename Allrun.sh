#!/bin/sh
module load gcc rarray boost openmpi
export NUM_OF_PROC=2
make clean
make parallel_run
