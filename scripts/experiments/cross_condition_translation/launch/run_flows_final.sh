#!/bin/bash
# Spec v4 E4.4: final runs (GPU). 10 tasks: index // 5 = model (fm_nl, fm_lin), index % 5 = seed. Args: k.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --job-name=e4_flow_final
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
K="${1:?k}"
MODELS=(${MODELS:-fm_nl fm_lin})   # e.g. MODELS="fm_shared" sbatch --array=0-4 ...
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 5))]}; SEED=$((SLURM_ARRAY_TASK_ID % 5))
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
echo "Starting ${SLURM_JOB_ID} ${MODEL} seed ${SEED} k ${K} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
  "source /ext3/env.sh; cd ${CONN2CONN_DIR}/scripts/experiments/cross_condition_translation; python -u run_flows.py final --model ${MODEL} --k ${K} --seed ${SEED}"
