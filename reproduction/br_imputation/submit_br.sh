#!/bin/bash
# One-command launch for the BR-only run (Glasser x 10 seeds). Submits the array, chains finalize
# to run ONLY if every unit succeeds (--dependency=afterok). Nothing to poll: watch sentinels.
#
#   cd /scratch/ans9868/Conn2Conn/reproduction/br_imputation && bash submit_br.sh
#
# Watch (no squeue polling):
#   ls sentinels/DONE_br_*.sentinel 2>/dev/null | grep -v finalize | wc -l   # /10
#   test -f sentinels/DONE_br_finalize.sentinel && echo FINALIZE DONE
#   ls -1t logs/br_*.txt | head -1 | xargs tail -40
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs sentinels outputs/parts

AID=$(sbatch --parsable run_br_unit.sbatch)
echo "submitted array job: $AID  (10 Glasser seeds, %10)"
FID=$(sbatch --parsable --dependency=afterok:"$AID" finalize_br.sbatch)
echo "submitted finalize  : $FID  (runs after array succeeds)"
echo
echo "watch (no squeue):"
echo "  ls sentinels/DONE_br_*.sentinel 2>/dev/null | grep -v finalize | wc -l   # /10"
echo "  test -f sentinels/DONE_br_finalize.sentinel && echo FINALIZE DONE"
