#!/bin/bash
# Latent-direct — one array task per Glasser seed = 3 units (seeds 0-2).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=lat_unit
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/latent_direct/logs/lat-%A_%a.out
# RESOURCES: per seed = 2 cross-modal arms x 5 objectives x (train+test+resid latents) + FCSC.
#   obj1a's 5-fold OOF per-PC BR is the driver (same as obj_functions). 16G ample; 4h generous.

set -u
ROOT=/scratch/ans9868/Conn2Conn
LD=$ROOT/reproduction/latent_direct
mkdir -p "$LD/outputs/parts/family" "$LD/logs" "$LD/sentinels"
cd "$LD"
SEED=$SLURM_ARRAY_TASK_ID
rm -f "$LD/sentinels/DONE_lat_${SEED}.sentinel" "$LD/sentinels/ERROR_lat_${SEED}.sentinel"
LOG="$LD/logs/lat_Glasser_s${SEED}.txt"
echo "lat unit SEED=$SEED" > "$LOG"

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
    cd $LD
    python run_latent_unit.py --seed $SEED >> $LOG 2>&1
" || { echo failed; touch "$LD/sentinels/ERROR_lat_${SEED}.sentinel"; exit 1; }
touch "$LD/sentinels/DONE_lat_${SEED}.sentinel"
echo "lat unit $SEED done"
