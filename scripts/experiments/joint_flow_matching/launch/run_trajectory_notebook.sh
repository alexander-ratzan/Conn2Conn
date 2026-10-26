#!/bin/bash
# Execute flow_trajectories.ipynb in place (E2.1 second addendum, seed 0). Submitted afterok on the seed-0 runs.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --job-name=e21_traj_nb
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
module purge
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
  "source /ext3/env.sh; python -I /scratch/asr655/tools/run_notebook.py /scratch/asr655/neuroinformatics/Conn2Conn/scripts/experiments/joint_flow_matching/flow_trajectories.ipynb"
