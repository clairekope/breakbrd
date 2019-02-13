#!/bin/bash
#SBATCH -N1 --exclusive --ntasks-per-node=14

module purge
module load slurm gcc openmpi python3

cd $SLURM_SUBMIT_DIR

mpirun python3 TNG_radial_bins.py

