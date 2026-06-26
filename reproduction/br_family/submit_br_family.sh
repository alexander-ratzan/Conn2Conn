#!/bin/bash
# One-command launch for the BR-family (F6/F7 heritability) run (Glasser x 10). Array + afterok
# finalize. Watch sentinels (no squeue).
#   cd /scratch/ans9868/Conn2Conn/reproduction/br_family && bash submit_br_family.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs sentinels outputs/parts
AID=$(sbatch --parsable run_br_family.sbatch)
echo "submitted array : $AID  (10 Glasser seeds, %10)"
FID=$(sbatch --parsable --dependency=afterok:"$AID" finalize_br_family.sbatch)
echo "submitted finalize: $FID  (afterok)"
echo
echo "watch (no squeue):"
echo "  ls sentinels/DONE_brfam_*.sentinel 2>/dev/null | grep -v finalize | wc -l   # /10"
echo "  test -f sentinels/DONE_brfam_finalize.sentinel && echo FINALIZE DONE"
