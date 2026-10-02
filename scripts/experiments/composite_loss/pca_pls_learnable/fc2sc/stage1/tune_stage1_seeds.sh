#!/bin/bash
# Composite-loss protocol, replicability instance, Stage 1 (spec v2 E1): 16-trial MSE-only tune (optimizer keys; architecture fixed) of CrossModal_PCA_PLS_learnable on
# SC -> FC, with varmatch / correye / neidist logged as monitor-only terms. Packed 4 trials per GPU with actor
# reuse (E0.6 pattern). Array index = shuffle seed (0-4).
#   sbatch scripts/experiments/composite_loss/pca_pls_learnable/fc2sc/stage1/tune_stage1_seeds.sh
# Needs the E1.1 loss code (loss_monitor_terms) on main.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=e1_stage1_pca_pls_learnable_fc2sc
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/e1_stage1_pca_pls_learnable_fc2sc_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/e1_stage1_pca_pls_learnable_fc2sc_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

SEED=${SLURM_ARRAY_TASK_ID}
CONFIG="scripts/experiments/composite_loss/pca_pls_learnable/fc2sc/stage1/CrossModal_PCA_PLS_learnable_mse.yml"

export RAY_worker_register_timeout_seconds=120
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Model=CrossModal_PCA_PLS_learnable  Source=SC  Config=${CONFIG}  Seed=${SEED}  (16 trials, 4 packed per GPU)"

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
      --model CrossModal_PCA_PLS_learnable \
      --config ${CONFIG} \
      --source FC \
      --target SC \
      --shuffle_seed ${SEED} \
      --data_load_mode precomputed \
      --save_checkpoint \
      --use_tune \
      --search_alg optuna \
      --num_samples 16 \
      --max_concurrent_trials 4 \
      --tune_cpus_per_trial 2 \
      --tune_gpus_per_trial 0.25 \
      --tune_reuse_actors true \
      --report_best_after_tune \
      --store_eval_md
  "

echo "Job Over at $(date)"
