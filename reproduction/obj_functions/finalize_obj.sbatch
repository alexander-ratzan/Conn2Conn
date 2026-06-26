#!/bin/bash
# Finalize objective-functions Phase 1 (afterok): 3-axis scorecard across seeds.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=obj_final
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/obj_functions/logs/obj-finalize-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
OBJ=$ROOT/reproduction/obj_functions
mkdir -p "$OBJ/logs" "$OBJ/sentinels"
cd "$OBJ"
rm -f "$OBJ/sentinels/DONE_obj_finalize.sentinel" "$OBJ/sentinels/ERROR_obj_finalize.sentinel"
LOG="$OBJ/logs/obj_finalize.txt"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export HOME=$ROOT/reproduction/outputs/wandb_home XDG_CACHE_HOME=$ROOT/reproduction/outputs/wandb_cache/xdg
    mkdir -p \$HOME \$XDG_CACHE_HOME
    cd $OBJ
    python finalize_obj.py > $LOG 2>&1
" || { echo failed; touch "$OBJ/sentinels/ERROR_obj_finalize.sentinel"; exit 1; }
touch "$OBJ/sentinels/DONE_obj_finalize.sentinel"
echo "obj finalize done"
