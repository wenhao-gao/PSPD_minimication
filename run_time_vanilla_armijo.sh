#!/bin/bash -l
#SBATCH --time=24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --partition=shared
#SBATCH -A t-jgray21
#SBATCH --reservation=Jgray319

module load python/3.7-anaconda
source activate myPyRosetta

cat task_time_vanilla_armijo.sh | /home-4/wgao12@jhu.edu/bin/parallel -j 6 --no-notice
