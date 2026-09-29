#!/bin/bash
# Array: 4 covariate conditions x 10 seeds = 40 tasks (0-39)
# task_id = seed_idx * 4 + cov_idx
#
# Conditions:
#   0: fs_all       - FreeSurfer all ROI features only
#   1: fs_volumes   - FreeSurfer whole-brain volume features only
#   2: demo         - age + sex + race_eth only
#   3: fs_all_demo  - fs_all + age + sex + race_eth
#
# This is a closed-form PCA-space model, so CPU-only resources are sufficient.

#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:30:00
#SBATCH --mem=32GB
#SBATCH --job-name=tune_cgauss_cov_array
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_cgauss_cov_array_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_cgauss_cov_array_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu
#SBATCH --array=0-39

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

export RAY_worker_register_timeout_seconds=120
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export TUNE_CPUS_PER_TRIAL=1
export TUNE_GPUS_PER_TRIAL=0
export MAX_CONCURRENT_TRIALS=4

COV_LABELS=("fs_all" "fs_volumes" "demo" "fs_all_demo")
CONFIGS=(
  "CrossModal_ConditionalGaussian_SC_fs_all.yml"
  "CrossModal_ConditionalGaussian_SC_fs_volumes.yml"
  "CrossModal_ConditionalGaussian_SC_demo.yml"
  "CrossModal_ConditionalGaussian_SC_fs_all_demo.yml"
)
NUM_COVS=4
SEEDS=(0 1 2 3 4 5 6 7 8 9)

IDX=${SLURM_ARRAY_TASK_ID}
COV_IDX=$((IDX % NUM_COVS))
SEED=${SEEDS[$((IDX / NUM_COVS))]}
COV_LABEL=${COV_LABELS[$COV_IDX]}
CONFIG=${CONFIGS[$COV_IDX]}

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "CovType=${COV_LABEL}  Config=${CONFIG}  Seed=${SEED}"

singularity exec \
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
      --model CrossModal_ConditionalGaussian \
      --config models/configs/${CONFIG} \
      --source SC \
      --target FC \
      --shuffle_seed ${SEED} \
      --data_load_mode precomputed \
      --save_checkpoint \
      --use_tune \
      --search_alg optuna \
      --num_samples 16 \
      --max_concurrent_trials ${MAX_CONCURRENT_TRIALS} \
      --tune_cpus_per_trial ${TUNE_CPUS_PER_TRIAL} \
      --tune_gpus_per_trial ${TUNE_GPUS_PER_TRIAL} \
      --report_best_after_tune \
      --store_eval_md
  "

echo "Job Over at $(date)"
