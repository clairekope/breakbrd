#!/bin/bash
#SBATCH -N1 --exclusive --ntasks-per-node=28

module purge
module load slurm gcc openmpi python3

cd $SLURM_SUBMIT_DIR

#echo "Starting Spectra"; mpirun python3 TNG_spectra.py 0.0 --dust --inst
echo "Measuring D4000";  python3 get_d4000.py 0.0 --dust --inst

