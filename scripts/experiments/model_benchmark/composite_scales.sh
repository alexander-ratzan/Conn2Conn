#!/bin/bash
# Spec v3 E3 / I1: reference scales for MODEL (default Sarwar2020MLP), array task i -> direction [sc2fc, fc2sc][i / 5], seed i % 5.
#   sbatch --export=ALL,MODEL=<Model> scripts/experiments/model_benchmark/composite_scales.sh   (then: composite_scales.py --combine)
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --job-name=e3_comp_scales
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
module purge
DIRS=(sc2fc fc2sc)
DIRECTION=${DIRS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}; SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))
MODEL="${MODEL:-Sarwar2020MLP}"
echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Campaign=e3_comp_scales Model=${MODEL} Direction=${DIRECTION} Seed=${SEED}"
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; export PYTHONUNBUFFERED=1 MPLBACKEND=Agg; cd /scratch/asr655/neuroinformatics/Conn2Conn; python scripts/experiments/model_benchmark/composite_scales.py --model ${MODEL} --direction ${DIRECTION} --seed ${SEED}"
echo "Job Over at $(date)"
