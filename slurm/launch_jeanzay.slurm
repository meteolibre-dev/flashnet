#!/bin/bash
#SBATCH --job-name=train_meteolibre
#SBATCH -C h100                   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # ou gpu:4 pour multi-GPU
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0   # ou ta version

cd $WORK/code/flashnet
python3 scripts/train_rf_mtg_lightning_shortcut_xpred.py
