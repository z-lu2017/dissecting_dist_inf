#!/bin/bash

#Submit this script with: sbatch run_experiment.slurm

#SBATCH --time=15:00:00                     # job time limit
#SBATCH --nodes=1                           # number of nodes
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task=1                   # number of CPU cores per task
#SBATCH --gres=gpu:1                        # gpu devices per node
#SBATCH --partition=gpu                     # partition
#SBATCH --mem=256G                          # memory (cpu)
#SBATCH --account=uvasrg_paid               # allocation name
#SBATCH -o slurm_results/run_experiment-%u-%j.out
#SBATCH -e slurm_results/run_experiment-%u-%j.err

# Activate venv
source /home/ujx4ab/ondemand/dissecting_dist_inf/.venv/bin/activate
cd /home/ujx4ab/ondemand/dissecting_dist_inf/experiments/

time python blackbox_attacks.py \
    --load_config "$ATTACK_CONFIG" \
    --en "$TEST_NAME"