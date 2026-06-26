#!/bin/bash
# Diagnostic: obj1a_ungated, one array task per Glasser seed = 5 units.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=diag_ung
#SBATCH --array=0-4%5
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/obj_functions/logs/diag-%A_%a.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
OBJ=$ROOT/reproduction/obj_functions
mkdir -p "$OBJ/outputs/parts" "$OBJ/logs" "$OBJ/sentinels"
cd "$OBJ"
SEED=$SLURM_ARRAY_TASK_ID
rm -f "$OBJ/sentinels/DONE_diag_${SEED}.sentinel" "$OBJ/sentinels/ERROR_diag_${SEED}.sentinel"
LOG="$OBJ/logs/diag_s${SEED}.txt"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    export HOME=$ROOT/reproduction/outputs/wandb_home XDG_CACHE_HOME=$ROOT/reproduction/outputs/wandb_cache/xdg
    mkdir -p \$HOME \$XDG_CACHE_HOME
    cd $OBJ
    python run_diag_ungated.py --seed $SEED > $LOG 2>&1
" || { echo failed; touch "$OBJ/sentinels/ERROR_diag_${SEED}.sentinel"; exit 1; }
touch "$OBJ/sentinels/DONE_diag_${SEED}.sentinel"
echo "diag unit $SEED done"
