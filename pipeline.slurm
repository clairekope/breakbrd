#!/bin/bash
#SBATCH -N1 --exclusive --ntasks-per-node=14

export PIPELINE_ARGS='0.0 --tng -l'

module purge
module load slurm gcc openmpi python3

cd $SLURM_SUBMIT_DIR

# The pipeline

#echo "Downloading FITS"; python3 download_fits.py $PIPELINE_ARGS
#echo "Downloading particles"; mpirun python3 download_cutouts.py $PIPELINE_ARGS
#echo "Analysing Gas"; mpirun python3 gas_analysis.py $PIPELINE_ARGS
#echo "Making Spectra"; mpirun python3 stellar_spectra.py $PIPELINE_ARGS
echo "Photometric Cuts"; mpirun python3 illustris_cuts.py $PIPELINE_ARGS
#echo "Measuring D4000";  python3 get_d4000.py $PIPELINE_ARGS

