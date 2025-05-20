#!/bin/bash

#Submit this script with: sbatch run_experiment.slurm

#SBATCH --time=10:00:00                     # job time limit
#SBATCH --nodes=1                           # number of nodes
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task=1                   # number of CPU cores per task
#SBATCH --gres=gpu:1                        # gpu devices per node
#SBATCH --partition=gpu                     # partition
#SBATCH --mem=32G                           # memory (cpu)
#SBATCH -J "Experiment (Bp_wind, 0.05, [0.0, 0.05, 0.1])"
#SBATCH --mail-user=ujx4ab@virginia.edu     # email address
#SBATCH --account=uvasrg_paid               # allocation name
#SBATCH -o slurm_results/run_experiment-%u-%j.out
#SBATCH -e slurm_results/run_experiment-%u-%j.err

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
export DATASET="lstm_wind_turbine"
export TRAIN_CONFIG="loss_and_threshold_attack.json"
export TEST_NAME="Bp_wind_0.5_[0.0_0.05_0.1]_Allvic_Alladv"

# Activate virtual environment
source /home/ujx4ab/ondemand/dissecting_dist_inf/.venv/bin/activate
cd experiments/

# Run your Python script
time python /home/ujx4ab/ondemand/dissecting_dist_inf/experiments/blackbox_attacks.py \
    --load_config /home/ujx4ab/ondemand/dissecting_dist_inf/experiments/configs/$DATASET/$TRAIN_CONFIG \
    --en $TEST_NAME