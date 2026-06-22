#!/bin/bash
# F8 PC-MECHANISM GRID — one array task per (parcellation x seed) = 20 units.
# Each unit: PCA SC/FC -> per-PC FC->R² + family AUC + sex/bv confound + PC1 resid + node strength
# -> per-unit npz in outputs/parts_f8/<parc>/. Cross-seed alignment + localization + verdict via
# finalize_f8.py. Same light profile as F6/F7 (16G / 8 CPU / 1h).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=f8_unit
#SBATCH --array=0-19%10
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/family_mechanism/logs/f8-%A_%a.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
FMD=$ROOT/reproduction/family_mechanism
mkdir -p "$FMD/outputs/parts_f8" "$FMD/logs" "$FMD/sentinels"
cd "$FMD"

PARCS=(Glasser 4S456Parcels)
IDX=$SLURM_ARRAY_TASK_ID
PARC=${PARCS[$((IDX/10))]}
SEED=$((IDX%10))
TAG="${PARC}_s${SEED}"
rm -f "$FMD/sentinels/DONE_f8_${IDX}.sentinel" "$FMD/sentinels/ERROR_f8_${IDX}.sentinel"
LOG="$FMD/logs/f8_${TAG}.txt"
echo "f8 unit IDX=$IDX PARC=$PARC SEED=$SEED" > "$LOG"

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
    echo '=== F8 PC mechanism ===' >> $LOG &&
    python run_f8_pcmech.py --parc $PARC --seed $SEED >> $LOG 2>&1
" || { echo failed; touch "$FMD/sentinels/ERROR_f8_${IDX}.sentinel"; exit 1; }
touch "$FMD/sentinels/DONE_f8_${IDX}.sentinel"
echo "f8 unit $IDX done"
