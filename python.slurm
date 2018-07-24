#!/bin/bash
#SBATCH -N1 --exclusive --ntasks-per-node=14

module purge
module load slurm gcc openmpi python3

cd $SLURM_SUBMIT_DIR

#mpirun python3 illustris_cuts.py
#mpirun python3 stellar_mass_growth.py
mpirun python3 gas_analysis.py
#mpirun python3 stellar_spectra.py
