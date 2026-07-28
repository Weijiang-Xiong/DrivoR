#!/bin/bash
#SBATCH --job-name RUN_EXP
#SBATCH --account=luts
#SBATCH --qos=normal
#SBATCH --partition=h100
#SBATCH --output="out/slurm-%j.log"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5800
#SBATCH --cpus-per-task=16

module load gcc/13.2.0 cuda/12.4.1
echo "${@:1}"
srun python -u "${@:1}"