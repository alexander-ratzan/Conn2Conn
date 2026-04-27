#!/bin/bash
# Array: 10 seeds for NodalMLP (SC source, all features on) (0-9)
# Runtime est: per-trial ~30s (ASHA early-stop) to ~20-30min (full 1000ep on H200/A100, ~45min-1h on L40S).
# Per-task (16 trials + best retrain + eval): ~1h30m (H200) / ~2h15m (A100) / ~2h30m-3h (L40S).
# Full 10-seed experiment: ~4-6h wall with typical ~7-GPU quota; ~2-3h if fully parallel.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=tune_nodalmlp_array
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_nodalmlp_array_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_nodalmlp_array_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu
#SBATCH --array=0-3

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

export RAY_worker_register_timeout_seconds=120

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export TUNE_CPUS_PER_TRIAL=4
export TUNE_GPUS_PER_TRIAL=1
export MAX_CONCURRENT_TRIALS=1

SEEDS=(0 1 2 3 4 5 6 7 8 9)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$IDX]}
CONFIG="NodalMLP_all_features.yml"

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Model=NodalMLP  Source=SC  Features=all  Config=${CONFIG}  Seed=${SEED}"

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
      --model NodalMLP \
      --config models/configs/${CONFIG} \
      --source SC \
      --target FC \
      --shuffle_seed ${SEED} \
      --data_load_mode precomputed \
      --save_checkpoint \
      --use_tune \
      --search_alg optuna \
      --num_samples 12 \
      --max_concurrent_trials ${MAX_CONCURRENT_TRIALS} \
      --tune_cpus_per_trial ${TUNE_CPUS_PER_TRIAL} \
      --tune_gpus_per_trial ${TUNE_GPUS_PER_TRIAL} \
      --report_best_after_tune \
      --store_eval_md
  "

echo "Job Over at $(date)"
