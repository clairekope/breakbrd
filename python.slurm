#!/bin/bash
#SBATCH -N1 --exclusive --ntasks-per-node=14

module purge
module load slurm gcc openmpi python3

cd $SLURM_SUBMIT_DIR

# The pipeline
#mpirun python3 illustris_cuts.py
mpirun python3 gas_analysis.py
mpirun python3 stellar_mass_growth.py
echo "Starting Spectra"
mpirun python3 stellar_spectra.py
python3 get_d4000.py

#mpirun python3 download_cutouts.py
