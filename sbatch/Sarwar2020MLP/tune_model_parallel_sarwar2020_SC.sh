#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=tune_sarwar_SC
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_sarwar_SC_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_sarwar_SC_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

export RAY_worker_register_timeout_seconds=120

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Heavier model than PCA-PLS-learnable, keep one GPU per trial.
# 4 GPUs allocated => run up to 4 concurrent trials.
export TUNE_CPUS_PER_TRIAL=4
export TUNE_GPUS_PER_TRIAL=1
export MAX_CONCURRENT_TRIALS=4

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"

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
      --model Sarwar2020MLP \
      --config models/configs/Sarwar2020MLP.yml \
      --source SC \
      --target FC \
      --shuffle_seed 0 \
      --save_checkpoint \
      --use_tune \
      --search_alg optuna \
      --num_samples 24 \
      --max_concurrent_trials ${MAX_CONCURRENT_TRIALS} \
      --tune_cpus_per_trial ${TUNE_CPUS_PER_TRIAL} \
      --tune_gpus_per_trial ${TUNE_GPUS_PER_TRIAL} \
      --report_best_after_tune \
      --store_eval_md
  "

echo "Job Over at $(date)"
