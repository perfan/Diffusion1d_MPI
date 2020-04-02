#!/bin/bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --job-name diffusin1dScaling
#SBATCH --output=run.log

module load gcc rarray boost openmpi

for NUM_OF_PROC in {1..32} 
do
    echo "Running Code over $NUM_OF_PROC cores"
    mpirun -n ${NUM_OF_PROC} ./diffusion1d params.ini
done
echo "All Done"
