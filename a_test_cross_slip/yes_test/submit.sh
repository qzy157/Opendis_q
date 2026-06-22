#!/bin/bash
#SBATCH --partition=comp
#SBATCH --job-name=load
#SBATCH --nodes=1           # Request 1 node
#SBATCH --ntasks=1          # Total number of tasks
#SBATCH --cpus-per-task=48  # Number of CPU cores per task
#SBATCH --time=48:00:00

module purge
module load miniforge/25.3.1
source activate opendis_cpu
python yes_test.py