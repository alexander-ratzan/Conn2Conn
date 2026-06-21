#!/bin/bash
# FINALIZE — runs after the unit array (chained via --dependency=afterany). Generates the
# completeness manifest, merges per-unit CSVs, verifies completeness (hard-fail on missing
# cells), global leak hard-fail, then summarizes into reports/reproduction_findings.md.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=grid_final
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/logs/finalize-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
RD=$ROOT/reproduction
cd "$RD"
mkdir -p sentinels
rm -f sentinels/DONE_finalize.sentinel sentinels/ERROR_finalize.sentinel
LOG="$RD/finalize_output.txt"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    cd $RD
    echo '=== gen_expected_cells ===' > $LOG
    python gen_expected_cells.py >> $LOG 2>&1 &&
    echo '=== merge_parts ===' >> $LOG &&
    python merge_parts.py >> $LOG 2>&1 &&
    echo '=== verify_completeness ===' >> $LOG &&
    python verify_completeness.py >> $LOG 2>&1
    VERIFY=\$?
    echo \"verify exit=\$VERIFY\" >> $LOG
    echo '=== leak check on MERGED downstream (authoritative) ===' >> $LOG
    python run_leak_checks.py --downstream-csv outputs/downstream.csv \
        --out outputs/leak_verdict.csv >> $LOG 2>&1
    LEAK=\$?
    echo \"leak exit=\$LEAK\" >> $LOG
    echo '=== summarize ===' >> $LOG &&
    python summarize.py >> $LOG 2>&1
    [ \$VERIFY -eq 0 ] && [ \$LEAK -eq 0 ]
" || { echo 'finalize: verify/leak failed (report still written)'; touch "$RD/sentinels/ERROR_finalize.sentinel"; }
touch "$RD/sentinels/DONE_finalize.sentinel"
echo "finalize done"
