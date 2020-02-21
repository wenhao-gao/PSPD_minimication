#!/bin/bash -l
#SBATCH --time=24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=shared
#SBATCH -A t-jgray21
#SBATCH --reservation=Jgray319

module load python/3.7-anaconda 
source activate myPyRosetta

start=$(date "+%s")
cat task_test.sh | /home-4/wgao12@jhu.edu/bin/parallel -j 1 --no-notice
now=$(date "+%s")

time=$((now-start))
echo "time used:$time seconds"
