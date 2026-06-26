#!/bin/bash
# Objective-functions Phase 1 — one array task per Glasser seed = 5 units (seeds 0-4).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=obj_unit
#SBATCH --array=0-4%5
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/obj_functions/logs/obj-%A_%a.out
# RESOURCES: per seed builds 5 estimators x (pred_te + pred_tr + pred_resid). obj1a adds 5-fold
#   OOF per-PC BR (~1500 small fits) -> the heavy one. 16G ample (PCA on 64,620-dim is the driver);
#   4h is generous for one seed.

set -u
ROOT=/scratch/ans9868/Conn2Conn
OBJ=$ROOT/reproduction/obj_functions
mkdir -p "$OBJ/outputs/parts/family" "$OBJ/logs" "$OBJ/sentinels"
cd "$OBJ"
SEED=$SLURM_ARRAY_TASK_ID
rm -f "$OBJ/sentinels/DONE_obj_${SEED}.sentinel" "$OBJ/sentinels/ERROR_obj_${SEED}.sentinel"
LOG="$OBJ/logs/obj_Glasser_s${SEED}.txt"
echo "obj unit SEED=$SEED" > "$LOG"

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
    python run_obj_unit.py --seed $SEED >> $LOG 2>&1
" || { echo failed; touch "$OBJ/sentinels/ERROR_obj_${SEED}.sentinel"; exit 1; }
touch "$OBJ/sentinels/DONE_obj_${SEED}.sentinel"
echo "obj unit $SEED done"
