#!/bin/bash
# H — per-subject achieved-vs-ceiling + reliability-filtered re-run (SC->FC, bv+demo->FC, 10 seeds)
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=noise_h
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/slurm-h-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
ND=$ROOT/notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check
cd "$ND"
rm -f DONE_h.sentinel ERROR_h.sentinel

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
    cd $ND
    python h_per_subject_and_filtered.py > h_per_subject_and_filtered_output.txt 2>&1
" || { echo failed; touch "$ND/ERROR_h.sentinel"; exit 1; }
touch "$ND/DONE_h.sentinel"
echo done
