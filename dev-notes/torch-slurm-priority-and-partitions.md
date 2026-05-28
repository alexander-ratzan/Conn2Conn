# SLURM submission strategy on Torch — account, partition, resource sizing

What we learned the hard way (2026-05-28) about making interactive jobs *actually start* on NYU Torch. Companion to [`torch-jupyter-tmux-skill.md`](torch-jupyter-tmux-skill.md), which covers the env stack (apptainer / conda / Jupyter); this file covers *how to get scheduled in the first place*.

---

## TL;DR — the one srun line to use

```bash
srun -A torch_pr_60_tandon_priority \
     --partition=cpu_short \
     --cpus-per-task=4 --mem=32G --time=04:00:00 \
     --pty bash
```

For a CPU-only interactive job (notebook, dev shell). Lands in **< 5 min** under typical cluster load. Two changes vs the project's older sbatch defaults:

1. **Account `_priority`** (not `_advanced`) — much higher FairShare per ans9868.
2. **`--partition=cpu_short` explicit** (not letting the scheduler auto-route to `all`).

GPU work is a different conversation — keep `-A torch_pr_60_tandon_advanced` and a GPU partition for those.

---

## Account selection (FairShare is the gate)

`ans9868` has access to three real submit accounts (plus `users`, which doesn't actually submit jobs):

| Account | FairShare | LevelFS | RawUsage | When to use |
|---|---|---|---|---|
| `torch_pr_60_tandon_priority` | **0.119** | **259** | 881 | **Default for CPU work.** Barely touched. Highest FairShare available. |
| `torch_pr_60_general` | 0.097 | 27.5 | 17,837 | Fallback for CPU work. Decent FairShare, mid-usage. |
| `torch_pr_60_tandon_advanced` | 0.055 | 7.4 | 136,537 | **Avoid for CPU.** Heavily over-shared (mostly by other people in the account). Reserved feel: keep for GPU runs where the `_advanced` QOS matters. |

**Counter-intuitive finding:** the names suggest `_priority` and `_advanced` are "fancy GPU-buy-in tiers" and `_general` is the default — wrong. `_priority` actually has higher SLURM-FairShare than `_general` *because it's been the least used*. The naming is misleading; trust the numbers.

### How to verify before submitting

```bash
sshare -U -u ans9868 -P -o "Account,FairShare,LevelFS,RawUsage"
```

Decision rule: pick the account with the **highest `FairShare`** value (this is what feeds directly into priority calc). `LevelFS` is the projected near-future fair-share factor and corroborates — > 1 = under-used and will be prioritized. If `_priority` ever rises to high usage (say RawUsage > 50,000), re-check whether `_general` has overtaken it.

### What FairShare buys you in absolute terms

Same job, two accounts, measured 2026-05-28:

| Account | Resulting job priority | Approx queue rank on `all` |
|---|---|---|
| `_advanced` | 11552 | rank 197+ (bottom) |
| `_priority` | 12192 | mid (still ~rank 100 on `all`) |

About **+640 priority points** for switching from `_advanced` to `_priority` — small in absolute terms but enough to leapfrog dozens of waiters at the boundary.

---

## Partition selection — *always* `--partition=cpu_short` for CPU work

If you omit `--partition`, the scheduler auto-routes. In practice it picks `all` (the cluster-wide pool) for most asks. **This is almost always the wrong choice for interactive notebooks.**

| Partition | Nodes | CPUs | RAM/CPU | Queue depth (typical) | Notes |
|---|---|---|---|---|---|
| `all` | 348 | 41,856 | 6.6 GB | ~200 pending | Cluster-wide, all node types incl. GPU. Most competition. |
| `cpu_short` | 254 | 32,512 | 4 GB | ~98 pending | The CPU-only sweet spot. Despite the name, `MaxTime=UNLIMITED` — there's no enforced walltime cap. |
| `cs` | 184 | 23,552 | 4 GB | similar | Bare-CPU subset of `cpu_short`. |
| `cpu_prem` | 348 | 41,856 | 6.6 GB | varies | Premium tier, otherwise like `all`. |

Why `cpu_short` wins for interactive notebooks:

- **Half the queue depth** of `all` — your priority places you much higher in the smaller pool.
- **Mostly CPU-only nodes** — the scheduler doesn't reserve GPU capacity, allocations resolve faster.
- **Backfill-friendly culture** — even without a hard `MaxTime`, the scheduler prefers shorter jobs here, so a 1-4 h walltime gets backfilled into idle slots fast.
- **No real downside** for 4 GB/CPU asks — only matters if you want > ~8 GB/CPU, which would push you toward `cpu_prem` / `all`.

---

## Resource sizing for this project

The Conn2Conn closed-form work is CPU-only (no GPU). Default sizing per session type:

| Workload | `--cpus-per-task` | `--mem` | `--time` | Notes |
|---|---|---|---|---|
| Interactive notebook (PCA/PLS/BR closed-form) | 4 | 32 GB | 04:00:00 | Comfortable headroom; lands in `cpu_short` fast |
| Same + 10-seed Exp 7 in [crossmodal_pca_pls_closed_form_overview](../notebooks-FC_to_SC-experimental/model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb) | 4 | 64 GB | 04:00:00 | Sim rebuilds across seeds may not GC tightly; bump mem |
| sbatch tune run (PCA/PLS family) | 4 | 64 GB | 01:00:00 | Matches existing `sbatch/CrossModal_PCA_PLS/` |
| sbatch tune run (Sarwar/Nodal MLP, GPU needed) | 4 | 96 GB | 04:00:00 + `--gres=gpu:1` | Use `_advanced` here, GPU partition |

**Do NOT request a GPU for the closed-form notebook.** It's pure CPU; a GPU request just slows queue time.

The "right size" rule of thumb: keep `mem / cpus` near the partition's per-CPU mean (4 GB/CPU on `cpu_short`, 6.6 GB/CPU on `all`/`cpu_prem`). Going much higher forces SLURM to find a node with mem headroom and slows scheduling slightly.

---

## When jobs still don't start — diagnostic playbook

If a job sits in `PENDING` longer than expected, **never poll `squeue` in a loop** (it spams the SLURM controller; HPC admins will email about it — see [memory feedback-torch-no-squeue-polling](../../../.claude/projects/-Users-user-projects-Conn2Conn/memory/feedback_torch_no_squeue_polling.md)). Make one-shot calls:

```bash
# 1. State + reason for one specific job
scontrol show job <jobid> | grep -E "JobState|Reason|Priority|StartTime|Partition|ReqTRES"

# 2. Estimated start time (often "N/A" when buried)
squeue -j <jobid> --start

# 3. Priority decomposition — see which factor is gating
sprio -j <jobid> -o "%i %Y %a %F %J %P %N"
#                    JobID Priority Age Fairshare JobSize Partition Nice

# 4. Where am I in the pending queue?
squeue -p <partition> -t PD -h -o "%Q %i %u" | sort -rn | awk -v me=<jobid> '
  {rank++; if ($2==me) {print "rank "rank; exit}}'

# 5. Snapshot fairshare for all my accounts (one-shot)
sshare -U -u ans9868 -P -o "Account,FairShare,LevelFS,RawUsage"
```

| Symptom | Likely cause | Fix |
|---|---|---|
| `StartTime=Unknown`, rank > 100 in pending queue | Auto-routed to `all`; not enough priority | Cancel + resubmit with `--partition=cpu_short` |
| `StartTime=Unknown`, low `FairShare` for current account | Account over-used | Cancel + resubmit with `-A torch_pr_60_tandon_priority` |
| `Reason=Resources` with idle CPUs visible | Memory shape doesn't fit available nodes | Lower `--mem` or move to higher RAM/CPU partition |
| `Reason=QOSMaxCPUPerUser` | Hit per-user CPU cap on this QOS | Wait for your own jobs to finish, or use a different QOS/account |
| `Reason=Priority` | Other jobs have higher priority | Wait for AGE to grow, or switch account |
| `Reason=BeginTime` | Job has a `--begin=` time set in the future | Usually not you; this is someone else's job ahead of you |
| `Reason=Dependency` | Job is waiting on another job to finish | Not your concern unless it's your own dependency |

---

## Two-jobs-in-parallel pattern (when you really want fast turnaround)

If you have spare fairshare headroom and the cluster is busy enough that one-shot waits are uncertain, submit two parallel jobs with **different shapes on different partitions** and take whichever lands first:

1. Job A: bigger ask on a wider partition (`-A _priority --partition=all --cpus-per-task=4 --mem=32G --time=02:00:00`)
2. Job B: smaller ask on a narrower partition (`-A _priority --partition=cpu_short --cpus-per-task=4 --mem=32G --time=04:00:00`)
3. Cancel the loser with `scancel <other-jobid>` the moment one lands — **never** `scancel -u $USER` (cancels all your jobs).

Each job's launch script should write to a **separate scratch dir** (e.g. `/scratch/ans9868/.jupyter_scratch/` vs `.jupyter_scratch_b/`) so the sentinel files (compute hostname, jupyter log, job id) don't race. Watch both sentinel pairs in one polling loop:

```bash
for tag in "" "_b"; do
  DIR=/scratch/ans9868/.jupyter_scratch${tag}
  if [ -s "$DIR/compute_host.txt" ] && grep -q "token=" "$DIR/jupyter.log"; then
    echo "WINNER=${tag:-A} HOSTNAME=$(cat $DIR/compute_host.txt) JOB=$(cat $DIR/job_id.txt)"
    break
  fi
done
```

This is overkill for normal workflows. Use only when (a) the cluster is unusually busy and (b) you have a hard deadline.

---

## Related

- Env stack on the compute node: [`torch-jupyter-tmux-skill.md`](torch-jupyter-tmux-skill.md)
- Project sbatch reference budgets: [`../sbatch/CrossModal_PCA_PLS/`](../sbatch/CrossModal_PCA_PLS/), [`../sbatch/Sarwar2020MLP/`](../sbatch/Sarwar2020MLP/) — these use the old `_advanced` account convention; new sbatch scripts should consider switching to `_priority` for CPU-only tune runs
- Memory: [feedback_torch_no_squeue_polling](../../../.claude/projects/-Users-user-projects-Conn2Conn/memory/feedback_torch_no_squeue_polling.md), [feedback_torch_sync_via_git](../../../.claude/projects/-Users-user-projects-Conn2Conn/memory/feedback_torch_sync_via_git.md), [reference_torch_conn2conn_paths](../../../.claude/projects/-Users-user-projects-Conn2Conn/memory/reference_torch_conn2conn_paths.md)
