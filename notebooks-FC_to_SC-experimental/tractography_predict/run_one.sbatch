#!/bin/bash
# Parameterized single-experiment runner for PARALLEL execution.
# Usage: sbatch run_one.sbatch <exp>   where <exp> in {e1,e2,e3,e4,e5,retest}
#
# Mounts the apptainer overlay READ-ONLY (:ro) so multiple jobs can run
# concurrently — apptainer refuses to open the same overlay :rw from two
# processes, which is what broke earlier parallel attempts. Our scripts only
# READ the conda env and WRITE results to /scratch (outside the overlay), so
# :ro is safe. PYTHONDONTWRITEBYTECODE stops Python writing .pyc into the
# now-read-only env.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=tp_one
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/tractography_predict/slurm-one-%j.out

set -u
EXP="${1:?usage: sbatch run_one.sbatch <e1|e2|e3|e4|e5|retest>}"
ROOT=/scratch/ans9868/Conn2Conn
TP=$ROOT/notebooks-FC_to_SC-experimental/tractography_predict
TRACT=$ROOT/notebooks-FC_to_SC-experimental/sanity_checks/tract_check

case "$EXP" in
  e1)     WD=$TP;    SCRIPT=e1_source_rep_comparison.py;      OUT=$TP/e1_source_rep_output.txt ;;
  e2)     WD=$TP;    SCRIPT=e2_asymmetry_across_reps.py;      OUT=$TP/e2_asymmetry_output.txt ;;
  e3)     WD=$TP;    SCRIPT=e3_marginal_r2t.py;               OUT=$TP/e3_marginal_output.txt ;;
  e4)     WD=$TP;    SCRIPT=e4_pc3_localization_on_r2t.py;    OUT=$TP/e4_localization_output.txt ;;
  e5)     WD=$TP;    SCRIPT=e5_downstream_cognition.py;       OUT=$TP/e5_downstream_output.txt ;;
  retest) WD=$TRACT; SCRIPT=retest_icc_pipeline.py;           OUT=$TRACT/retest_icc_output.txt ;;
  *) echo "unknown exp: $EXP"; exit 2 ;;
esac

rm -f "$TP/DONE_${EXP}.sentinel" "$TP/ERROR_${EXP}.sentinel"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=16
    export OPENBLAS_NUM_THREADS=16
    export MKL_NUM_THREADS=16
    cd $WD
    set -e
    echo '[run_one:$EXP] starting $SCRIPT ...'
    time python $SCRIPT > $OUT 2>&1
    echo '[run_one:$EXP] finished cleanly.'
" || {
    echo "[run_one:$EXP] FAILED"
    touch "$TP/ERROR_${EXP}.sentinel"
    exit 1
}
touch "$TP/DONE_${EXP}.sentinel"
echo "[run_one:$EXP] DONE_${EXP}.sentinel written"
