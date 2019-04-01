#!/bin/bash

#SBATCH -J deriv
#SBATCH -N 1
#SBATCH -p debug
#SBATCH -t 00:10:00
#SBATCH -o ./log_slurm/derivatives.o%j
#SBATCH -L cscratch1
#SBATCH -C haswell
#SBATCH -D /global/homes/c/chmodi/Programs/21cm/21cmhod/code
#SBATCH -A m3127
#DW jobdw capacity=10000GB access_mode=striped type=scratch
#DW stage_in source=/global/cscratch1/sd/chmodi/m3127/H1mass/highres/10240-9100-fixed destination=$DW_JOB_STRIPED/ type=directory

module unload darshan
module unload python

module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/5.3.0

source /usr/common/contrib/bccp/conda-activate.sh 3.6
export OMP_NUM_THREADS=1

bcast-pip -U --no-deps https://github.com/bccp/nbodykit/archive/master.zip
#bcast-pip -U --no-deps /global/homes/c/chmodi/Programs/nbodykit

echo 'Finally starting'
echo 'THis works, right?'
echo $DW_JOB_STRIPED

time srun -n 32 python -u derivatives.py $DW_JOB_STRIPED -p alpha 

#time srun -n 512 python -u derivatives.py mcut -d -0.05
#time srun -n 512 python -u derivatives.py alpha -d -0.05
#time srun -n 512 python -u derivatives.py alpha -s big -d -0.05
#time srun -n 512 python -u derivatives.py mcut -s big -d -0.05
#


##srun -N ${SLURM_NNODES} --ntasks-per-node 48 -c 4 ../codes/40eae2464/src/fastpm hod-desi.lua $DW_JOB_STRIPED


