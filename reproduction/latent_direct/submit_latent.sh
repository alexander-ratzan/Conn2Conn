#!/bin/bash
# One-command launch: 3-seed array + chained finalize (afterok). Sentinel-based; no squeue.
#   cd /scratch/ans9868/Conn2Conn/reproduction/latent_direct && bash submit_latent.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs sentinels outputs/parts/family
AID=$(sbatch --parsable run_latent_unit.sbatch); echo "array: $AID (seeds 0-2)"
FID=$(sbatch --parsable --dependency=afterok:"$AID" finalize_latent.sbatch); echo "finalize: $FID"
