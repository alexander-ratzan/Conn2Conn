#!/bin/bash
# Launch objective-functions Phase 1 (Glasser x 5 seeds): array + afterok finalize.
#   cd /scratch/ans9868/Conn2Conn/reproduction/obj_functions && bash submit_obj.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs sentinels outputs/parts/family
AID=$(sbatch --parsable run_obj_unit.sbatch)
echo "array    : $AID  (5 Glasser seeds)"
FID=$(sbatch --parsable --dependency=afterok:"$AID" finalize_obj.sbatch)
echo "finalize : $FID  (afterok)"
echo "watch: ls sentinels/DONE_obj_*.sentinel | grep -v finalize | wc -l  # /5"
