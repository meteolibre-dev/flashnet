#!/bin/bash
#SBATCH --job-name=train_meteolibre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # ou gpu:4 pour multi-GPU
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=@gpu   # ou @v100, @a100

module purge
module load pytorch-gpu/py3/2.9.0   # ou ta version

srun python train.py --args
