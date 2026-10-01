#!/bin/bash
# Krakencoder composite-loss grid: one array task = PER_JOB fits trained concurrently on one GPU, then scored.
#   python scripts/experiments/composite_loss/krakencoder/grid_runner.py plan --set <SET>   # prints the --array range
#   sbatch --array=0-<n-1> --export=ALL,SET=pilot scripts/experiments/composite_loss/krakencoder/launch_grid.sh
# Optional: EXTRA="--epochs 20 --checkpoint-every 10 --tag-suffix _smoke" for checks.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=160GB
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=cl_kraken
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/cl_kraken_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/cl_kraken_%A_%a.err

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

SET=${SET:-pilot}
PER_JOB=${PER_JOB:-4}
EXTRA=${EXTRA:-}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export EVAL_THREADS=8

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)  SET=${SET} PER_JOB=${PER_JOB} ${EXTRA}"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 \
  > "${CONN2CONN_DIR}/results/logs/cl_kraken_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

singularity exec --nv \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1
    cd ${CONN2CONN_DIR}
    python scripts/experiments/composite_loss/krakencoder/grid_runner.py run --set ${SET} --per-job ${PER_JOB} ${EXTRA}
  "

echo "Job Over at $(date)"
