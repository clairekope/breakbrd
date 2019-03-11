#!/bin/bash
#SBATCH -N1 --exclusive --ntasks-per-node=28

module purge
module load slurm gcc openmpi python3

cd $SLURM_SUBMIT_DIR

mpirun python3 TNG_radial_bins.py 0.0 --dust --inst --parent

