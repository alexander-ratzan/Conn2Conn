#!/bin/bash
# Spec v3 E1.0 + E1.1 (CPU): the E1 acceptance checks, then the unnormalised Glasser SC cache build + cache gate.
#   sbatch scripts/experiments/covariate_ablation/prep.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=e1_cov_prep
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
module purge
REPO=/scratch/asr655/neuroinformatics/Conn2Conn
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MPLBACKEND=Agg
echo "Starting ${SLURM_JOB_ID} on $(hostname) at $(date); HEAD $(git -C ${REPO} rev-parse --short HEAD)"
run() { singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc "source /ext3/env.sh; cd ${REPO}; $1"; }
status=0
run "python scripts/experiments/covariate_ablation/checks/check_e1.py" || status=1
run "python scripts/experiments/covariate_ablation/build_sc_cache.py --workers ${SLURM_CPUS_PER_TASK:-8}" || status=1
echo "Job Over at $(date) (status ${status})"
exit ${status}
