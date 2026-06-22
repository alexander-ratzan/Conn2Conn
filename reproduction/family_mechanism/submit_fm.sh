#!/bin/bash
# One-command launch for the F6/F7 family-structure grid on Torch.
# Submits the 20-unit array, then chains finalize to run ONLY if every unit succeeds
# (--dependency=afterok). Nothing to poll: watch the sentinels under sentinels/.
#
#   cd /scratch/ans9868/Conn2Conn/reproduction/family_mechanism && bash submit_fm.sh
#
# Progress (passive, no squeue):
#   ls sentinels/DONE_fm_*.sentinel | wc -l        # units done (expect 20)
#   cat sentinels/DONE_fm_finalize.sentinel        # finalize done
#   tail outputs/family_auc.csv ; cat logs/fm_finalize.txt
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs sentinels outputs/parts

AID=$(sbatch --parsable run_fm_unit.sbatch)
echo "submitted array job: $AID  (20 units, %10 concurrency)"
FID=$(sbatch --parsable --dependency=afterok:"$AID" finalize_fm.sbatch)
echo "submitted finalize  : $FID  (runs after array succeeds)"
echo
echo "watch (no squeue polling):"
echo "  ls sentinels/DONE_fm_*.sentinel 2>/dev/null | grep -v finalize | wc -l   # /20"
echo "  test -f sentinels/DONE_fm_finalize.sentinel && echo FINALIZE DONE"
