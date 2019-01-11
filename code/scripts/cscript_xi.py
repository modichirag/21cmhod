#!/bin/bash

#SBATCH -J xi
#SBATCH -N 8
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -o ./log_slurm/xi.o%j
#SBATCH -L cscratch1
#SBATCH -C haswell
#SBATCH -D /global/homes/c/chmodi/Programs/21cm/21cmhod/code
#SBATCH -A m3127

# Load NbodyKit
#
#source /usr/common/contrib/bccp/conda-activate.sh 3.6
#bcast-pip https://github.com/bccp/nbodykit/archive/master.zip

module unload python
module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/5.3.0

#module load python/3.6-anaconda
#export PATH=/global/homes/c/chmodi/.conda/envs/fastpm/bin/:$PATH

source /usr/common/contrib/bccp/conda-activate.sh 3.6
bcast-pip -U --no-deps https://github.com/bccp/nbodykit/archive/master.zip
#bcast-pip -U --no-deps https://github.com/rainwoodman/nbodykit/archive/store-catalog.zip
#bcast-pip  -U --no-deps https://github.com/bccp/simplehod/archive/master.zip

export OMP_NUM_THREADS=1
echo 'Finally starting'

time srun -n 256 python -u xi.py
