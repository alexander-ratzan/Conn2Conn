#!/bin/bash
# F6/F7 FAMILY-STRUCTURE GRID — one array task per (parcellation x seed) = 20 units.
# Each unit: build 8 connectome variants for this seed's test split, extract per-pair
# demeaned-cosine sims bucketed by relation -> per-unit npz in outputs/parts/<parc>/.
# Pooled aggregation (AUC + bootstrap + perm + FDR) runs once via finalize_fm.py.
# Reuses the main grid's container/env/profile (24G / 4h / 8 CPU; see run_unit.sbatch notes).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=fm_unit
#SBATCH --array=0-19%6
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/family_mechanism/logs/fm-%A_%a.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
FMD=$ROOT/reproduction/family_mechanism
mkdir -p "$FMD/outputs/parts" "$FMD/logs" "$FMD/sentinels"
cd "$FMD"

PARCS=(Glasser 4S456Parcels)
IDX=$SLURM_ARRAY_TASK_ID
PARC=${PARCS[$((IDX/10))]}
SEED=$((IDX%10))
TAG="${PARC}_s${SEED}"
rm -f "$FMD/sentinels/DONE_fm_${IDX}.sentinel" "$FMD/sentinels/ERROR_fm_${IDX}.sentinel"
LOG="$FMD/logs/fm_${TAG}.txt"
echo "fm unit IDX=$IDX PARC=$PARC SEED=$SEED" > "$LOG"

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
    cd $FMD
    echo '=== F6/F7 family pair sims ===' >> $LOG &&
    python run_f6_family.py --parc $PARC --seed $SEED >> $LOG 2>&1
" || { echo failed; touch "$FMD/sentinels/ERROR_fm_${IDX}.sentinel"; exit 1; }
touch "$FMD/sentinels/DONE_fm_${IDX}.sentinel"
echo "fm unit $IDX done"
