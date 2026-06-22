#!/bin/bash
# F6/F7 FINALIZE — pool all per-unit family caches per parcellation and aggregate.
# Chained after run_fm_unit.sbatch via --dependency=afterok (see submit_fm.sh), so it only
# starts once all 20 units succeed. Runs finalize_fm.py in the container: pool seed npz ->
# AUC + bootstrap + perm + FDR -> outputs/family_auc.csv (+ Glasser-vs-notebook regression guard).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=fm_finalize
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/family_mechanism/logs/fm_finalize-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
FMD=$ROOT/reproduction/family_mechanism
mkdir -p "$FMD/logs" "$FMD/sentinels"
cd "$FMD"
LOG="$FMD/logs/fm_finalize.txt"
rm -f "$FMD/sentinels/DONE_fm_finalize.sentinel" "$FMD/sentinels/ERROR_fm_finalize.sentinel"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    cd $FMD
    python finalize_fm.py > $LOG 2>&1
" || { echo failed; touch "$FMD/sentinels/ERROR_fm_finalize.sentinel"; exit 1; }
touch "$FMD/sentinels/DONE_fm_finalize.sentinel"
echo "fm finalize done"
