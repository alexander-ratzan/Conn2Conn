#!/bin/bash
# Spec v4 E2.1 addendum: synthetic check of the un-anchored flows (reported, not a gate).
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=e21_add_synth
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
  "source /ext3/env.sh; cd ${CONN2CONN_DIR}; python -u scripts/experiments/joint_flow_matching/checks/synthetic_e21_addendum.py"
