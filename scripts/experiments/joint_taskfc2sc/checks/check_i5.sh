#!/bin/bash
# Spec v3 I5 acceptance (CPU): check_i5.py (A-D) and the before / after fingerprint (E, HEAD worktree).
#   sbatch scripts/experiments/joint_taskfc2sc/checks/check_i5.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=e2_check_i5
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
module purge
REPO=/scratch/asr655/neuroinformatics/Conn2Conn
OLD=/scratch/asr655/neuroinformatics/Conn2Conn_wt_i5base
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MPLBACKEND=Agg
echo "Starting ${SLURM_JOB_ID} on $(hostname) at $(date); HEAD $(git -C ${REPO} rev-parse --short HEAD)"
[ -d "${OLD}" ] || git -C "${REPO}" worktree add --detach "${OLD}" HEAD
run() { singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc "source /ext3/env.sh; cd ${REPO}; $1"; }
run "python scripts/experiments/joint_taskfc2sc/checks/fingerprint_fc2sc.py ${OLD}" | sed 's/^/before I5: /'
run "python scripts/experiments/joint_taskfc2sc/checks/fingerprint_fc2sc.py ${REPO}" | sed 's/^/after I5:  /'
git -C "${REPO}" worktree remove --force "${OLD}"
run "python scripts/experiments/joint_taskfc2sc/checks/check_i5.py"
echo "Job Over at $(date)"
