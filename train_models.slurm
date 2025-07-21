#!/bin/bash

#Submit this script with: sbatch train_models.slurm

#SBATCH --time=15:00:00                     # job time limit
#SBATCH --nodes=1                           # number of nodes
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task=1                   # number of CPU cores per task
#SBATCH --gres=gpu:1                        # gpu devices per node
#SBATCH --partition=gpu                     # partition
#SBATCH --constraint="a6000|v100|a100|a40"
#SBATCH --mem=128G                          # memory (cpu)
#SBATCH --account=uvasrg_paid               # allocation name
#SBATCH -o slurm_results/train_models-%u-%j.out
#SBATCH -e slurm_results/train_models-%u-%j.err

# Activate venv, replace source path
source /home/ujx4ab/ondemand/dissecting_dist_inf/.venv/bin/activate
module load cuda/12.4.1
cd /home/ujx4ab/ondemand/dissecting_dist_inf/experiments/

# Run training
time python train_models.py \
    --load_config "$TRAIN_CONFIG" \
    --gpu 0