#!/bin/bash
#SBATCH -N 2
#SBATCH -n 128
#SBATCH --job-name='GV_ent_sm'
#SBATCH -t 168:00:00
#SBATCH --mem-per-cpu=2GB 

export PIPELINE_ARGS='0.0 --tng -m -l/mnt/research/galaxies-REU/sims/IllustrisTNG'
#export PIPELINE_ARGS='0.5 --tng -l -m'
#export PIPELINE_ARGS='0.1 --tng -l -m'
#export PIPELINE_ARGS='0.03 --tng -m -l/mnt/research/galaxies-REU/sims/TNG003'
#export PIPELINE_ARGS='0.0 -l -m'

#module purge
#module load slurm gcc openmpi python3

cd $SLURM_SUBMIT_DIR

# The pipeline
#echo "Downloading FITS"; python3 download_fits.py $PIPELINE_ARGS
#echo "Downloading particles"; mpirun python3 download_cutouts.py $PIPELINE_ARGS
#echo "Getting Particle Info"; mpirun python3 particle_info.py $PIPELINE_ARGS
#echo "Making Spectra"; mpirun python3 stellar_spectra.py $PIPELINE_ARGS
#echo "Measuring Disk Color"; mpirun python3 disk_color.py $PIPELINE_ARGS
#echo "Measuring D4000";  python3 get_d4000.py $PIPELINE_ARGS
#echo "Getting Densities"; mpirun python3 galaxy_density.py $PIPELINE_ARGS
#mpirun python3 alt_galaxy_environ.py $PIPELINE_ARGS  
#mpirun python3 particle_info_tracked.py $PIPELINE_ARGS
#mpirun python3 entropy_profiles.py $PIPELINE_ARGS
mpirun python3 green_valley_properties.py $PIPELINE_ARGS
