#!/bin/bash
# G — per-subject reliability distribution (tiny: load cached cells, vectorized cosine).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=noise_g
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/slurm-g-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
ND=$ROOT/notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check
cd "$ND"
rm -f DONE_g.sentinel ERROR_g.sentinel

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    cd $ND
    python g_per_subject_reliability.py > g_per_subject_reliability_output.txt 2>&1
" || { echo failed; touch "$ND/ERROR_g.sentinel"; exit 1; }
touch "$ND/DONE_g.sentinel"
echo done
