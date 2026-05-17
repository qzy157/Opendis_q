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

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# 从最新 restart 续跑：传入 step id，例如 63 → 读 output/restart.63.exadis
# 若 output/ 为空（首次运行），不带参数从 ../relax_low_t/output/config.90000.data 起跑
python low_t.py