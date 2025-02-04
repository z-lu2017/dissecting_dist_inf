#!/bin/bash

#Submit this script with: sbatch train_models.slurm

#SBATCH --time=10:00:00                     # job time limit
#SBATCH --nodes=1                           # number of nodes
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task=1                   # number of CPU cores per task
#SBATCH --gres=gpu:1                        # gpu devices per node
#SBATCH --partition=gpu                     # partition
#SBATCH --mem=32G                           # memory (cpu)
#SBATCH -J "Train (new_census/race/vic/0.2/250)"
#SBATCH --mail-user=ujx4ab@virginia.edu     # email address
#SBATCH --account=uvasrg_paid               # allocation name
#SBATCH -o slurm_results/train_models-%u-%j.out
#SBATCH -e slurm_results/train_models-%u-%j.err

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
export DATASET="new_census"
export TRAIN_CONFIG="train_models.json"

# Activate virtual environment
source /home/ujx4ab/ondemand/dissecting_dist_inf/.venv/bin/activate

# Make sure the cuda is loaded
module load cuda/12.4.1

# Run your Python script
time python /home/ujx4ab/ondemand/dissecting_dist_inf/experiments/train_models.py \
  --load_config /home/ujx4ab/ondemand/dissecting_dist_inf/experiments/configs/$DATASET/$TRAIN_CONFIG \
  --gpu 0