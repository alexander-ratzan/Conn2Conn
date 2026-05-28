# Skill: Reliable Torch Notebook Workflow (tmux + SSH tunnel + $SCRATCH)

This document is the standard operating procedure for running Jupyter notebooks on NYU Torch without VS Code Remote-SSH.

## Goal

Run Jupyter on a compute node (not login node), keep all runtime/cache writes in `$SCRATCH`, and access notebooks from your laptop browser via SSH port forwarding.

## Prerequisites

- You can SSH from laptop with:
  - `ssh torch`
- You can request a compute allocation via Slurm (`srun`)
- Your container/env paths exist:
  - Overlay: `/scratch/ans9868/kraken_env/unlocked_kraken_env.ext3`
  - Image: `/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`
  - Conda env: `/ext3/miniforge3/envs/kraken_env`

## One-time local SSH config (laptop)

Edit `/Volumes/CrucialX6/Home/.ssh/config` and ensure this host exists:

```sshconfig
Host torch
    HostName login.torch.hpc.nyu.edu
    User ans9868
    ServerAliveInterval 60
    ServerAliveCountMax 10
    ConnectTimeout 60
```

## Daily Runbook

### 1) Login and start tmux on Torch

```bash
ssh torch
tmux new -s nb
```

If session already exists:

```bash
tmux attach -t nb
```

### 2) Move to a compute node (never run heavy work on login node)

```bash
srun -A torch_pr_60_tandon_priority \
     --partition=cpu_short \
     --cpus-per-task=4 --mem=32G --time=04:00:00 \
     --pty bash
hostname
```

Expected: hostname like `cs608.hpc.nyu.edu`, not `torch-login-...`

> Account + partition + sizing is load-bearing — picking the wrong account or letting SLURM auto-route to partition `all` can leave you stuck in the queue for 30+ min. See [torch-slurm-priority-and-partitions.md](torch-slurm-priority-and-partitions.md) for why we use `_priority` over `_advanced` and explicit `--partition=cpu_short` over auto-routing, plus the resource-sizing table for this project's workloads.

### 3) Start container + env

```bash
apptainer shell --fakeroot --overlay /scratch/ans9868/kraken_env/unlocked_kraken_env.ext3:rw /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
source /ext3/miniforge3/bin/activate
conda activate /ext3/miniforge3/envs/kraken_env
export PYTHONNOUSERSITE=True
```

### 4) Force all Jupyter/cache/runtime writes to `$SCRATCH`

```bash
export SCR_BASE=/scratch/ans9868/.jupyter_scratch
mkdir -p $SCR_BASE/{jupyter,runtime,config,cache,ipython,matplotlib,tmp}

export HOME=$SCR_BASE
export XDG_RUNTIME_DIR=$SCR_BASE/runtime
export XDG_CACHE_HOME=$SCR_BASE/cache
export XDG_CONFIG_HOME=$SCR_BASE/config
export JUPYTER_CONFIG_DIR=$SCR_BASE/jupyter
export JUPYTER_DATA_DIR=$SCR_BASE/jupyter
export JUPYTER_RUNTIME_DIR=$SCR_BASE/runtime
export IPYTHONDIR=$SCR_BASE/ipython
export MPLCONFIGDIR=$SCR_BASE/matplotlib
export TMPDIR=$SCR_BASE/tmp
```

### 5) Launch Jupyter on compute node

```bash
jupyter lab --no-browser --allow-root --ip=0.0.0.0 --port=8889
```

Important:

- `--allow-root` is required in this container workflow.
- `--ip=0.0.0.0` is required so login-node tunnel can reach the compute node service.

### 6) From laptop, open tunnel to compute node

In a second local terminal:

```bash
ssh -N -L 8889:cs603.hpc.nyu.edu:8889 torch
```

Replace `cs603.hpc.nyu.edu` with the actual compute hostname from step 2.

### 7) Open notebook in browser (laptop)

Use local URL with token printed by Jupyter:

```text
http://127.0.0.1:8889/lab?token=<TOKEN>
```

Keep both terminals open:

- HPC/tmux window running Jupyter
- Laptop tunnel window running `ssh -N -L ...`

## Quick Health Checks

### Check Jupyter listen address on compute node

```bash
ss -lntp | grep 8889
```

Healthy for this workflow:

- `0.0.0.0:8889` (reachable via tunnel from login node)

### Check local tunnel is listening on laptop

```bash
lsof -iTCP:8889 -sTCP:LISTEN -n -P
```

Healthy:

- `ssh` listening on `127.0.0.1:8889`

## Cleanup / stop

- Stop notebook: `Ctrl+C` in Jupyter terminal
- Detach tmux: `Ctrl+b`, then `d`
- Reattach later: `tmux attach -t nb`

---

# Problems We Hit and Why

## 1) Host key mismatch errors

Symptoms:

- `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!`

Cause:

- Stale/changed host fingerprints in `known_hosts`.

Fix:

- Remove stale entries with `ssh-keygen -R <host>` and reconnect to accept new host key.

## 2) `ssh-copy-id` to gateway failed after MFA

Symptoms:

- Gateway login succeeded but then restricted-shell/home errors.

Cause:

- Gateway policy/shell restrictions; not a reliable place for key setup in this flow.

Fix:

- Move to direct Torch auth flow and stop relying on gateway for VS Code-style setup.

## 3) VS Code Remote-SSH timed out after successful auth

Symptoms:

- Authenticated, then `Connecting with SSH timed out`.

Cause:

- Tight extension timeouts + multi-stage auth + cluster restrictions.

Fix:

- Switched to robust terminal + tmux + browser notebook workflow.

## 4) Quota exceeded at `/root/.local/share`

Symptoms:

- `OSError: [Errno 122] Disk quota exceeded: '/root/.local/share'`

Cause:

- Jupyter runtime/cache defaulted to root-like paths inside container.

Fix:

- Redirected HOME/XDG/Jupyter/IPython/TMP paths to `$SCRATCH`.

## 5) `Connection refused` on tunnel even though Jupyter was “running”

Symptoms:

- `channel 2: open failed: connect failed: Connection refused`

Causes:

- Jupyter bound only to `127.0.0.1` on compute node
- Tunnel initially targeted login node loopback instead of compute host

Fixes:

- Start Jupyter with `--ip=0.0.0.0`
- Tunnel to compute hostname: `-L 8889:<compute-host>:8889`

## 6) Job/step already completing or completed

Symptoms:

- `srun: error: Unable to create step ... already completing or completed`

Cause:

- Allocation/session lifecycle race (step ended while trying to re-enter).

Fix:

- Re-enter active compute node (`ssh <node>`) or request a new `srun` allocation.

## 7) Repeated MFA prompts / “logged in twice” feeling

Cause:

- Multiple SSH channels/sessions in some client workflows.

Fix:

- Expected behavior; keep one stable tunnel and one stable Jupyter process in tmux.

---

# Recommended Defaults Going Forward

- Keep notebook workflow terminal-first on Torch.
- Use `tmux` so work survives disconnects.
- Keep all notebook caches/runtime under `/scratch/ans9868/.jupyter_scratch`.
- Use Git for source sync (`push` local, `pull` on Torch).

