#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
# 2 GPUs: 1 per trial, max_concurrent_trials=2 → 2 trials in parallel
#SBATCH --gres=gpu:2
#SBATCH --job-name=tune_proj_SC_demo_fs
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_proj_SC_demo_fs_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_proj_SC_demo_fs_%j.err
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

export TUNE_CPUS_PER_TRIAL=3
export TUNE_GPUS_PER_TRIAL=1
export MAX_CONCURRENT_TRIALS=2

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
      --model CrossModal_PCA_PLS_CovProjector \
      --config models/configs/CrossModal_PCA_PLS_CovProjector_SC_demo_fs.yml \
      --save_checkpoint \
      --use_tune \
      --source SC \
      --target FC \
      --num_samples 24 \
      --max_concurrent_trials ${MAX_CONCURRENT_TRIALS} \
      --tune_cpus_per_trial ${TUNE_CPUS_PER_TRIAL} \
      --tune_gpus_per_trial ${TUNE_GPUS_PER_TRIAL} \
      --report_best_after_tune \
      --store_eval_md
  "

echo "Job Over at $(date)"
