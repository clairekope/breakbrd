#!/bin/bash
#SBATCH -N1 --exclusive --ntasks-per-node=14

module purge
module load slurm gcc openmpi python3

cd $SLURM_SUBMIT_DIR

# The pipeline
#python3 download_fits.py
mpirun python3 illustris_cuts.py # generated parent cut but not g-r; on hold til I check FITs units
#mpirun python3 download_cutouts.py # download in progress
#mpirun python3 gas_analysis.py
#mpirun python3 stellar_mass_growth.py # needs correction for new z
#echo "Starting Spectra"
#mpirun python3 stellar_spectra.py
#python3 get_d4000.py

