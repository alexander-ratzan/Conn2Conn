#!/bin/bash
# Prune Ray storage under results/ while keeping every run's logs (spec v3 backlog: artifact cleanup / storage).
#
#   ray_tmp/ray_<jobid>/session_*/   per-SLURM-job Ray session. Keeps `logs/` (driver, raylet, worker stdout/stderr:
#                                    a few MB) of EVERY session; removes the rest (`artifacts/` trial-output copies,
#                                    `runtime_resources/`, sockets, port files) for sessions that are
#                                      - older than KEEP_DAYS (newest file in the session),
#                                      - not from a job still in squeue,
#                                      - not listed in KEEP_SESSIONS (full reference copies of working runs),
#                                      - not the weekly reference: for every ISO week before KEEP_DAYS, the newest
#                                        session whose SLURM job COMPLETED (else the newest session) stays whole.
#   ray_checkpoints/<Model>_tune_*/  Tune storage. Keeps result.json / progress.csv / params / events / error.txt of
#                                    every trial; removes `checkpoint_*` dirs (model weights) in tune dirs older than
#                                    CKPT_DAYS. `best/` (closed-form best configs) is never touched. Best-trial
#                                    reports and their retrained checkpoints live in results/ray_results/ (untouched).
#
# Dry run by default (prints what it would remove and how much); `--apply` deletes.
#   bash scripts/maintenance/prune_ray_storage.sh            # dry run
#   bash scripts/maintenance/prune_ray_storage.sh --apply
set -euo pipefail

APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
RESULTS="${RESULTS:-/scratch/asr655/neuroinformatics/Conn2Conn/results}"
KEEP_DAYS="${KEEP_DAYS:-21}"          # ray_tmp sessions newer than this stay whole
CKPT_DAYS="${CKPT_DAYS:-90}"          # ray_checkpoints tune dirs newer than this keep their checkpoints
# Full reference sessions of working runs (2026-10-20): E2.2 launch_model.sh tune (E3's pipeline), v2:E1 stage-1
# composite-loss tunes (linear backbone, both directions), the actor-reuse preflight.
KEEP_SESSIONS="${KEEP_SESSIONS:-ray_19248105 ray_18900301 ray_19053917 ray_18871767}"

now=$(date +%s)
running=" $(squeue -u "$USER" -h -o %A 2>/dev/null | sort -u | tr '\n' ' ') "
total_kb=0

remove() {  # remove() <path>: size it, then delete if --apply
  local kb; kb=$(du -sk "$1" 2>/dev/null | cut -f1); total_kb=$((total_kb + ${kb:-0}))
  printf "%10s KB  %s\n" "${kb:-0}" "$1"
  if [ "$APPLY" = 1 ]; then rm -rf -- "$1"; fi
}

echo "== ray_tmp (keep logs/ everywhere; strip sessions older than ${KEEP_DAYS} d, not running, not reference)"
# Session table: name, newest file time, ISO week, SLURM state (COMPLETED ranks first within a week).
table=$(mktemp)
for s in "$RESULTS"/ray_tmp/ray_*; do
  [ -d "$s" ] || continue
  name=$(basename "$s")
  newest=$(find "$s" -maxdepth 4 -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  echo "$name ${newest:-$now} $(date -d @"${newest:-$now}" +%G-W%V)"
done > "$table"
ids=$(awk '{sub(/^ray_/, "", $1); print $1}' "$table" | paste -sd,)
states=$(sacct -j "$ids" -X -n -P -o JobIDRaw,State 2>/dev/null || true)
weekly=" $(while read -r name ts week; do
  st=$(grep -m1 "^${name#ray_}|" <<< "$states" | cut -d'|' -f2)
  [ $(( (now - ts) / 86400 )) -lt "$KEEP_DAYS" ] && continue
  echo "$week $([ "$st" = COMPLETED ] && echo 1 || echo 0) $ts $name"
done < "$table" | sort -k1,1 -k2,2nr -k3,3nr | awk '!seen[$1]++ {print $4}' | tr '\n' ' ') "
rm -f "$table"
echo "weekly references kept whole:${weekly}"
for s in "$RESULTS"/ray_tmp/ray_*; do
  [ -d "$s" ] || continue
  name=$(basename "$s"); id=${name#ray_}
  [[ "$running" == *" $id "* ]] && continue
  [[ " $KEEP_SESSIONS " == *" $name "* ]] && continue
  [[ "$weekly" == *" $name "* ]] && continue
  newest=$(find "$s" -maxdepth 4 -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  [ $(( (now - ${newest:-$now}) / 86400 )) -lt "$KEEP_DAYS" ] && continue
  for sess in "$s"/session_2*; do
    [ -d "$sess" ] || continue
    for item in "$sess"/*; do
      [ "$(basename "$item")" = "logs" ] && continue
      remove "$item"
    done
  done
done
tmp_kb=$total_kb

echo "== ray_checkpoints (keep trial metadata; remove checkpoint_* in tune dirs older than ${CKPT_DAYS} d)"
cutoff=$(( now - CKPT_DAYS * 86400 ))
for t in "$RESULTS"/ray_checkpoints/*_tune_*; do
  [ -d "$t" ] || continue
  [ "$(stat -c %Y "$t")" -ge "$cutoff" ] && continue
  while IFS= read -r c; do remove "$c"; done < <(find "$t" -type d -name 'checkpoint_*' -prune)
done

printf "\nray_tmp: %.1f GB   ray_checkpoints: %.1f GB   total: %.1f GB   (%s)\n" \
  "$(echo "$tmp_kb / 1048576" | bc -l)" "$(echo "($total_kb - $tmp_kb) / 1048576" | bc -l)" \
  "$(echo "$total_kb / 1048576" | bc -l)" "$([ "$APPLY" = 1 ] && echo removed || echo 'dry run; --apply to delete')"
