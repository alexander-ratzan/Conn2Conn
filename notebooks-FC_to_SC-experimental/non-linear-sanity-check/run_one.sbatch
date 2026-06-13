#!/bin/bash
# Parameterized single-experiment runner for the non-linear-sanity-check suite.
# Usage: sbatch run_one.sbatch <n1|n2|n3>
# Mounts the apptainer overlay :ro so jobs run concurrently (4 cpu each to fit the
# 32-core per-user QOS cap). PYTHONDONTWRITEBYTECODE so Python doesn't write into ro env.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=nl_one
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/non-linear-sanity-check/slurm-one-%j.out

set -u
EXP="${1:?usage: sbatch run_one.sbatch <n1|n2|n3>}"
ROOT=/scratch/ans9868/Conn2Conn
NL=$ROOT/notebooks-FC_to_SC-experimental/non-linear-sanity-check

case "$EXP" in
  n1) SCRIPT=n1_nonlinear_cognition.py;       OUT=$NL/n1_cognition_output.txt ;;
  n2) SCRIPT=n2_nonlinear_reconstruction.py;  OUT=$NL/n2_reconstruction_output.txt ;;
  n3) SCRIPT=n3_nonlinear_marginal.py;        OUT=$NL/n3_marginal_output.txt ;;
  *) echo "unknown exp: $EXP"; exit 2 ;;
esac

rm -f "$NL/DONE_${EXP}.sentinel" "$NL/ERROR_${EXP}.sentinel"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    cd $NL
    set -e
    echo '[nl:$EXP] starting $SCRIPT ...'
    time python $SCRIPT > $OUT 2>&1
    echo '[nl:$EXP] finished cleanly.'
" || { echo "[nl:$EXP] FAILED"; touch "$NL/ERROR_${EXP}.sentinel"; exit 1; }
touch "$NL/DONE_${EXP}.sentinel"
echo "[nl:$EXP] DONE_${EXP}.sentinel written"
