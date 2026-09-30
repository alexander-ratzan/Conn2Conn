#!/bin/bash
# reuse_actors preflight (spec v2 E0.6): the same 12-trial NodalMLP bilinear tune, packed 4 trials per GPU,
# run with Ray Tune actor reuse on (task 0) and off (task 1). W&B is offline so these trials never reach the
# E0 scrape. Measures: trial errors, dataset (HCP_Base) builds, one timed HCP_Base build, trial and job wall
# time, and GPU utilization.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=preflight_reuse_actors
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/preflight_reuse_actors_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/preflight_reuse_actors_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

REUSE_OPTIONS=(true false)
REUSE=${REUSE_OPTIONS[$SLURM_ARRAY_TASK_ID]}
CONFIG="scripts/experiments/nodal_models_benchmark/preflight/NodalMLP_bilinear_preflight.yml"
GPU_LOG="${CONN2CONN_DIR}/results/logs/preflight_reuse_actors_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv"

export RAY_worker_register_timeout_seconds=120
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}, reuse_actors=${REUSE}) on $(hostname) at $(date)"

# GPU utilization every 15 s for the whole job
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 15 > "${GPU_LOG}" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

JOB_START=$(date +%s)
singularity exec --nv \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --env "RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1
    export WANDB_MODE=offline
    unset RAY_TMPDIR
    cd ${CONN2CONN_DIR}
    python - <<'PY'
import time
from data.hcp_dataset import HCP_Base
t = time.time()
HCP_Base(parcellation='Glasser', hemi='both', shuffle_seed=0, source='SC', target='FC', cov_sources=['fs_all'],
         expose_node_features=True, expose_sc_matrix=True, data_load_mode='precomputed')
print(f'[preflight] one HCP_Base build: {time.time() - t:.1f} s', flush=True)
PY
    python main.py \
      --mode prod \
      --model NodalMLP \
      --config ${CONFIG} \
      --source SC \
      --target FC \
      --shuffle_seed 0 \
      --data_load_mode precomputed \
      --use_tune \
      --search_alg optuna \
      --num_samples 12 \
      --max_concurrent_trials 4 \
      --tune_cpus_per_trial 2 \
      --tune_gpus_per_trial 0.25 \
      --tune_reuse_actors ${REUSE}
  "

echo "[preflight] reuse_actors=${REUSE} job wall time: $(( $(date +%s) - JOB_START )) s"
echo "Job Over at $(date)"
