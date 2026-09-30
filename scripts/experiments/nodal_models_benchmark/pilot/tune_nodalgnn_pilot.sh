#!/bin/bash
# NodalGNN pilot (spec v2 E0.5, D2): 12-trial MSE-only tune on SC -> FC, packed 4 trials per GPU with actor reuse
# (E0.6). Array index = shuffle seed. Pilot: --array=0 (default). If seed 0's best val_demeaned_r >= 0.045, run
# seeds 1-3 with:  sbatch --array=1-3 scripts/experiments/nodal_models_benchmark/pilot/tune_nodalgnn_pilot.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=tune_nodalgnn_pilot
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_nodalgnn_pilot_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_nodalgnn_pilot_%A_%a.err
#SBATCH --array=0

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

SEED=${SLURM_ARRAY_TASK_ID}
CONFIG="scripts/experiments/nodal_models_benchmark/pilot/NodalGNN_mse_pilot.yml"

export RAY_worker_register_timeout_seconds=120
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Model=NodalGNN  Source=SC  Config=${CONFIG}  Seed=${SEED}  (12 trials, 4 packed per GPU)"

singularity exec --nv \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --env "RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1
    unset RAY_TMPDIR
    cd ${CONN2CONN_DIR}
    python main.py \
      --mode prod \
      --model NodalGNN \
      --config ${CONFIG} \
      --source SC \
      --target FC \
      --shuffle_seed ${SEED} \
      --data_load_mode precomputed \
      --save_checkpoint \
      --use_tune \
      --search_alg optuna \
      --num_samples 12 \
      --max_concurrent_trials 4 \
      --tune_cpus_per_trial 2 \
      --tune_gpus_per_trial 0.25 \
      --tune_reuse_actors true \
      --report_best_after_tune \
      --store_eval_md
  "

echo "Job Over at $(date)"
