#!/bin/bash
# One-command launch for the F8 PC-mechanism grid on Torch.
# Submits the 20-unit array, chains finalize via --dependency=afterok. Sentinel-based; no squeue.
#
#   cd /scratch/ans9868/Conn2Conn/reproduction/family_mechanism && bash submit_f8.sh
#
# Progress:
#   ls sentinels/DONE_f8_*.sentinel 2>/dev/null | grep -v finalize | wc -l   # /20
#   test -f sentinels/DONE_f8_finalize.sentinel && echo FINALIZE DONE
#   cat logs/f8_finalize.txt ; head outputs/f8_per_pc.csv
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs sentinels outputs/parts_f8

AID=$(sbatch --parsable run_f8_unit.sbatch)
echo "submitted F8 array job: $AID  (20 units, %10)"
FID=$(sbatch --parsable --dependency=afterok:"$AID" finalize_f8.sbatch)
echo "submitted F8 finalize : $FID  (runs after array succeeds)"
