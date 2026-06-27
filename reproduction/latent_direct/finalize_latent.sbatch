#!/bin/bash
# Latent-direct finalize — pool seeds -> scorecard_latent.csv + Δ vs round-trip (chained afterok).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=lat_finalize
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/latent_direct/logs/lat_finalize-%j.out
set -u
ROOT=/scratch/ans9868/Conn2Conn
LD=$ROOT/reproduction/latent_direct
cd "$LD"; mkdir -p logs sentinels
rm -f sentinels/DONE_lat_finalize.sentinel sentinels/ERROR_lat_finalize.sentinel
apptainer exec --fakeroot --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif bash -lc "
    source /ext3/miniforge3/bin/activate && conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True OMP_NUM_THREADS=2
    cd $LD && python finalize_latent.py > logs/lat_finalize.txt 2>&1
" || { touch sentinels/ERROR_lat_finalize.sentinel; exit 1; }
touch sentinels/DONE_lat_finalize.sentinel
