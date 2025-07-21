#!/usr/bin/env bash
#
# submit_all.sh  — submit one SLURM job per attack‑config

# path to config folder
CONFIG_DIR="/home/ujx4ab/ondemand/dissecting_dist_inf/experiments/configs/lstm_WT_new/active_experiment_scripts"

# slurm script that is run
SLURM_SCRIPT="/home/ujx4ab/ondemand/dissecting_dist_inf/run_experiment.slurm"

# snapshot diretory
SNAP_ROOT="${CONFIG_DIR}/.snaps"
mkdir -p "$SNAP_ROOT"

for cfg in "$CONFIG_DIR"/*.json; do
  base="$(basename "$cfg" .json)"
  snap="${SNAP_ROOT}/${base}.json"

  # copy to per‐job snapshot
  cp "$cfg" "$snap"

  TEST_NAME="$base"

  sbatch \
    -J "$TEST_NAME" \
    --export=ALL,ATTACK_CONFIG="$snap",TEST_NAME="$TEST_NAME" \
    "$SLURM_SCRIPT"

  echo "→ submitted $base as $TEST_NAME  (using $snap)"
done
