#!/bin/bash

#SBATCH -J 21cmhod
#SBATCH -N 16
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -o ./log_slurm/getbias.o%j
#SBATCH -L cscratch1
#SBATCH -C haswell
#SBATCH -D /global/homes/c/chmodi/Programs/21cm/21cmhod/code

module unload darshan
module unload python

module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/5.3.0

source /usr/common/contrib/bccp/conda-activate.sh 3.6
export OMP_NUM_THREADS=1

bcast-pip -U --no-deps https://github.com/bccp/nbodykit/archive/master.zip
#bcast-pip -U --no-deps /global/homes/c/chmodi/Programs/nbodykit

echo 'Finally starting'

#time srun -n 128 python -u distributeHI.py -m ModelA -s small
#time srun -n 128 python -u distributeHI.py -m ModelB -s small
#time srun -n 512 python -u distributeHI.py -m ModelC -s small
##time srun -n 128 python -u get_pks.py -m ModelA -s small
##time srun -n 128 python -u get_pks.py -m ModelB -s small
time srun -n 512 python -u get_pks.py -m ModelC -s small
##
#time srun -n 1024 python -u distributeHI.py -m ModelA -s big
#time srun -n 1024 python -u distributeHI.py -m ModelB -s big
#time srun -n 512 python -u distributeHI.py -m ModelC -s big
time srun -n 512 python -u get_pks.py -m ModelC -s big

time srun -n 512 python -u get_pks.py -m ModelA -s big
time srun -n 512 python -u get_pks.py -m ModelB -s big
#
