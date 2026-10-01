#!/bin/bash
# Retrain Krakencoder per seed (vendored upstream, spec v2 E2) and evaluate both directions through main.py.
# Array index = shuffle seed. Recipe and output tag come from CONFIG (default models/configs/Krakencoder.yml).
#   sbatch scripts/sbatch/Krakencoder/train_array_krakencoder_seeds.sh                      # seeds 0-9
#   sbatch --array=0 --export=ALL,CONFIG=<variant.yml>,EVAL_MODE=dev scripts/sbatch/Krakencoder/train_array_krakencoder_seeds.sh
# EVAL_MODE=prod (default) logs the two evaluation runs to W&B; dev evaluates without W&B.
# ~50 min per seed for the default recipe (2000 epochs, 4 flavors) on one GPU.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=train_krakencoder
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/train_krakencoder_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/train_krakencoder_%A_%a.err
#SBATCH --array=0-9

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

SEED=${SLURM_ARRAY_TASK_ID}
CONFIG=${CONFIG:-models/configs/Krakencoder.yml}
EVAL_MODE=${EVAL_MODE:-prod}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Model=Krakencoder  Config=${CONFIG}  Seed=${SEED}  Eval=${EVAL_MODE}"

singularity exec --nv \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1
    cd ${CONN2CONN_DIR}
    python scripts/krakencoder/train_krakencoder.py --config ${CONFIG} --seed ${SEED}
    for DIR in 'SC FC' 'FC SC'; do
      set -- \${DIR}
      python main.py --mode ${EVAL_MODE} --model Krakencoder --config ${CONFIG} \
        --source \$1 --target \$2 --shuffle_seed ${SEED} --data_load_mode precomputed
    done
  "

echo "Job Over at $(date)"
