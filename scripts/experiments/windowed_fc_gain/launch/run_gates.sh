#!/bin/bash
# windowed_fc_gain gates 1-6, one condition per array task (CPU). Then the report:
#   jid=$(sbatch --parsable scripts/experiments/windowed_fc_gain/launch/run_gates.sh)
#   sbatch --dependency=afterok:$jid scripts/experiments/windowed_fc_gain/launch/report.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=1:30:00
#SBATCH --array=0-7
#SBATCH --job-name=wfg_gates
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
CONDITIONS=(rest emotion gambling language motor relational social wm)
COND="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
module purge
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; cd ${CONN2CONN_DIR}; python -u scripts/experiments/windowed_fc_gain/run_gates.py --condition ${COND}"
