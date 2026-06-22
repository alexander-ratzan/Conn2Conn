#!/bin/bash
# F8 FINALIZE — cross-seed alignment + PC3 localization + verdict (chained afterok via submit_f8.sh).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=f8_finalize
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/family_mechanism/logs/f8_finalize-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
FMD=$ROOT/reproduction/family_mechanism
mkdir -p "$FMD/logs" "$FMD/sentinels"
cd "$FMD"
LOG="$FMD/logs/f8_finalize.txt"
rm -f "$FMD/sentinels/DONE_f8_finalize.sentinel" "$FMD/sentinels/ERROR_f8_finalize.sentinel"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    cd $FMD
    python finalize_f8.py > $LOG 2>&1
" || { echo failed; touch "$FMD/sentinels/ERROR_f8_finalize.sentinel"; exit 1; }
touch "$FMD/sentinels/DONE_f8_finalize.sentinel"
echo "f8 finalize done"
