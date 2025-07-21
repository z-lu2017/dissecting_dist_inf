#!/usr/bin/env bash
#
# submit_all_trainings.sh
# snapshots each train JSON and submit one SLURM job per config

# path to config folder
CONFIG_DIR="/home/ujx4ab/ondemand/dissecting_dist_inf/experiments/configs/lstm_WT_new/active_training_scripts"

# slurm script that is run
SLURM_SCRIPT="/home/ujx4ab/ondemand/dissecting_dist_inf/train_models.slurm"

# snap shot location
SNAP_ROOT="${CONFIG_DIR}/.snaps"
mkdir -p "$SNAP_ROOT"

for cfg in "$CONFIG_DIR"/*.json; do
  base="$(basename "$cfg" .json)"
  snap="${SNAP_ROOT}/${base}.json"

  cp "$cfg" "$snap"

  TRAIN_NAME="$base"

  sbatch \
    -J "$TRAIN_NAME" \
    --export=ALL,TRAIN_CONFIG="$snap",TRAIN_NAME="$TRAIN_NAME" \
    "$SLURM_SCRIPT"

  echo "→ submitted training '$TRAIN_NAME' using snapshot '$snap'"
done