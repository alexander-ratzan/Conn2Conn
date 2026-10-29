#!/bin/bash
# Spec v4 E4.3: flow search on seed 0 (GPU). 8 tasks: index // 4 = model (fm_nl, fm_lin), index % 4 = worker; workers of a
# model share one Optuna journal study. Args: k [trials per worker, default 8].
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --job-name=e4_flow_search
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
K="${1:?k}"; TRIALS="${2:-8}"
MODELS=(${MODELS:-fm_nl fm_lin})   # e.g. MODELS="fm_shared" sbatch --array=0-3 ...
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}; WORKER=$((SLURM_ARRAY_TASK_ID % 4))
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
echo "Starting ${SLURM_JOB_ID} ${MODEL} worker ${WORKER} k ${K} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
  "source /ext3/env.sh; cd ${CONN2CONN_DIR}/scripts/experiments/cross_condition_translation; python -u run_flows.py search --model ${MODEL} --k ${K} --trials ${TRIALS} --worker ${WORKER}"
