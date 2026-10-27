# HCP1200 xcp-d Data Transfer / Merge --- Handoff

## Goal

We are transferring a large HCP1200 xcp-d dataset from an external Mac
drive to NYU Torch HPC and **merging it into an existing partially
populated dataset**.

The critical safety requirement is:

> **Never overwrite, modify, or delete any file that already exists in
> the final HCP1200 destination. Only add files that are currently
> missing.**

The source and destination directory structures are intentionally
aligned. We have now successfully transferred and extracted the full
source dataset into a staging directory on Torch.

We are at the **final merge stage**.

------------------------------------------------------------------------

## 1. HPC environment

User:

``` text
asr655
```

Project/account:

``` text
torch_pr_59_tandon_advanced
```

Main project root:

``` text
/scratch/asr655/neuroinformatics/GeneEx2Conn_data/
```

### Staging / transferred dataset

``` text
/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200_transfer/
```

Contains:

``` text
xcpd-0-9-1.tar
xcpd-0-9-1/
untar.sh
untar_18967573.out
untar_18967573.err
```

The extracted source tree is:

``` bash
SRC_ROOT=/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200_transfer/xcpd-0-9-1
```

### Existing final destination

``` bash
DST_ROOT=/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_fMRI/xcpd-0-9-1
```

Both are under `/scratch` and are expected to be on the same filesystem.
This should be explicitly verified before relying on rename semantics,
e.g. using `stat -c %d`.

------------------------------------------------------------------------

## 2. Original source dataset

Original dataset was on a Mac external ExFAT drive:

``` text
/Volumes/MyDisk_1/xcpd-0-9-1/
```

It contains:

``` text
582,068 files
3,288 directories
~441–442 GiB logical/apparent data
```

On ExFAT, `du -sh` reported about `568G`, while apparent size was
\~`442G`. The discrepancy was due to ExFAT allocation overhead from
hundreds of thousands of small files.

------------------------------------------------------------------------

## 3. Tar creation and transfer

Because Globus/direct rsync of \~582k individual files was extremely
slow, the entire dataset was packed into an **uncompressed tar** on the
Mac:

``` bash
cd /Volumes/MyDisk_1
caffeinate tar -cf xcpd-0-9-1.tar xcpd-0-9-1/
```

Result:

``` text
/Volumes/MyDisk_1/xcpd-0-9-1.tar
```

Exact size:

``` text
473975155712 bytes
```

approximately 441 GiB.

The tar was transferred through Globus to:

``` text
/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200_transfer/xcpd-0-9-1.tar
```

Globus integrity verification was enabled.

Exact HPC tar size:

``` bash
stat -c '%s bytes' xcpd-0-9-1.tar
```

returned:

``` text
473975155712 bytes
```

Exact Mac source tar size:

``` bash
stat -f '%z bytes' /Volumes/MyDisk_1/xcpd-0-9-1.tar
```

also returned:

``` text
473975155712 bytes
```

Thus source and destination archive byte counts matched exactly, in
addition to Globus verification.

------------------------------------------------------------------------

## 4. Extraction on Torch

Extraction was performed as a Slurm CPU job rather than on a login node.

Job:

``` text
18967573
```

Slurm account:

``` text
torch_pr_59_tandon_advanced
```

The job ran on:

``` text
cs606.hpc.nyu.edu
```

and took:

``` text
01:31:27
```

Slurm reported the main job and batch step as `COMPLETED 0:0`.

The extraction script ended:

``` text
Extraction completed successfully.
Finished: Thu Oct 1 15:29:42 EDT 2026
```

There was also a `18967573.0 bash CANCELLED` entry. This was **only an
interactive `srun --jobid=18967573 --overlap --pty bash` shell**
attached for monitoring. Slurm terminated that shell when the main batch
job finished. It was not an extraction failure.

------------------------------------------------------------------------

## 5. Extracted dataset verification

After extraction:

``` text
du -sh:                 442G
du -sh --apparent-size: 441G
files:                   582068
directories:             3288
```

The file and directory counts exactly match the known source counts.

Therefore the extracted staging tree is considered successfully
transferred and extracted.

**Do not delete the tar yet.**

The original tar remains available as a ground-truth backup:

``` text
/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200_transfer/xcpd-0-9-1.tar
```

------------------------------------------------------------------------

## 6. Existing final destination is only partially populated

The existing destination is:

``` text
/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_fMRI/xcpd-0-9-1/
```

Current destination:

``` text
35 GiB apparent size
11,746 files
```

Compare with staging:

``` text
Staging:      582,068 files / ~441 GiB
Destination:   11,746 files / ~35 GiB
```

Therefore almost all staging data needs to be added.

------------------------------------------------------------------------

## 7. Important structure of the partial destination

The destination is **not a random partial copy**.

It intentionally contains a small subset of xcp-d products, primarily
existing resting-state functional connectivity matrices.

For example:

``` text
DST_ROOT/sub-100206/func/
```

contains files such as:

``` text
sub-100206_task-rest_dir-LR_run-1_space-fsLR_seg-4S456Parcels_stat-pearsoncorrelation_relmat.tsv
sub-100206_task-rest_dir-LR_run-1_space-fsLR_seg-Glasser_stat-pearsoncorrelation_relmat.tsv
sub-100206_task-rest_dir-LR_run-2_space-fsLR_seg-4S456Parcels_stat-pearsoncorrelation_relmat.tsv
sub-100206_task-rest_dir-LR_run-2_space-fsLR_seg-Glasser_stat-pearsoncorrelation_relmat.tsv
sub-100206_task-rest_space-fsLR_seg-4S456Parcels_stat-mean_timeseries.tsv
sub-100206_task-rest_space-fsLR_seg-4S456Parcels_stat-pearsoncorrelation_relmat.tsv
sub-100206_task-rest_space-fsLR_seg-Glasser_stat-pearsoncorrelation_relmat.tsv
```

There may also be empty aligned directories, e.g.:

``` text
DST_ROOT/sub-100206/anat/
```

was empty.

Meanwhile staging has the full xcp-d products for the same subject. For
example:

``` text
SRC_ROOT/sub-100206/func/
```

contains many task products (emotion, rest, ALFF, ReHo, coverage,
timeseries, correlation matrices, JSON sidecars, etc.) as well as copies
of some resting-state files already present in the destination.

The directory structures are intentionally aligned:

``` text
xcpd-0-9-1/
    sub-XXXXXX/
        func/
        ...
```

------------------------------------------------------------------------

## 8. Absolute safety requirement

The final operation must have these semantics:

``` text
For each staging file:

if destination path does not exist:
    add/move source file into destination
else:
    DO NOTHING
```

### Allowed

``` text
SRC/foo/bar.tsv exists
DST/foo/bar.tsv does not exist

=> move/add bar.tsv to DST
```

### Forbidden

``` text
SRC/foo/bar.tsv exists
DST/foo/bar.tsv exists

=> NEVER replace DST/foo/bar.tsv
```

Even if staging and existing copies are likely identical, **existing
destination files are authoritative and must remain byte-for-byte
untouched**.

Also:

-   Never delete destination files.
-   Never replace destination directories destructively.
-   Never use `--delete`.
-   Never blindly `mv` an existing subject directory onto another
    existing subject directory.
-   Do not remove the `.tar` until final verification is complete.

------------------------------------------------------------------------

## 9. Why we moved away from whole-tree rsync

Initially the proposed merge was:

``` bash
rsync -ah --ignore-existing SRC/ DST/
```

This is safe, but likely inefficient here.

Because the destination has only 11,746 files / 35 GiB while staging has
582,068 files / 441 GiB, rsync would physically copy roughly **406 GiB /
\~570k files** from one `/scratch` location to another.

Since staging and destination appear to be on the same filesystem, we
should be able to exploit **filesystem rename operations** instead.

A same-filesystem:

``` bash
mv source/file destination/file
```

should normally call `rename()` rather than physically read and rewrite
the file contents.

Thus hundreds of GB can potentially be reorganized primarily through
metadata operations.

This should be substantially faster than rsync/cp.

------------------------------------------------------------------------

## 10. Current proposed strategy

We are **not yet performing the full merge**.

The immediate next step is a **single-subject smoke test using
`sub-100206`**.

Source:

``` bash
SRC=/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200_transfer/xcpd-0-9-1/sub-100206
```

Destination:

``` bash
DST=/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_fMRI/xcpd-0-9-1/sub-100206
```

Before doing anything, verify source and destination really are on the
same filesystem/device:

``` bash
stat -c '%d %n' "$SRC" "$DST"
```

The numeric device IDs should match.

If they do not match, **do not use the rename optimization**.

------------------------------------------------------------------------

## 11. Proposed smoke-test move algorithm

The current proposed shell algorithm is file-level no-clobber moving:

``` bash
find "$SRC" -type f -print0 |
while IFS= read -r -d '' f; do

    rel="${f#"$SRC"/}"
    dest="$DST/$rel"

    if [[ ! -e "$dest" ]]; then
        mkdir -p "$(dirname "$dest")"
        mv -n "$f" "$dest"
    fi

done
```

The intention is:

``` text
destination absent
    -> mkdir parent if needed
    -> same-filesystem mv/rename

destination exists
    -> leave source copy where it is
    -> leave destination completely untouched
```

`mv -n` provides a second no-clobber defense in addition to the
existence check.

However, the coding agent should carefully review GNU `mv -n` semantics
on Torch and preferably design the operation so that **destination
existence is never ambiguous**.

The agent should prioritize correctness over shaving a few seconds.

------------------------------------------------------------------------

## 12. Useful property of this approach

After moving all *noncolliding* files from staging into destination,
staging naturally becomes a record of collisions.

For `sub-100206`, after a successful move:

``` bash
find "$SRC" -type f
```

should return primarily/only files whose corresponding destination paths
already existed.

Those should be the original rsFC matrices and any other pre-existing
products.

This gives a useful audit mechanism:

``` text
staging leftovers == paths deliberately not overwritten
```

The original tar still contains the complete source tree, so moving
files out of the extracted staging tree does not destroy the only copy.

------------------------------------------------------------------------

## 13. Smoke-test verification should be meticulous

Before modifying `sub-100206`, create a manifest of destination files.

At minimum:

``` bash
find "$DST" -type f -printf '%P\n' | sort > sub-100206_before_paths.txt
```

Preferably also checksum the small set of existing destination files:

``` bash
find "$DST" -type f -print0 |
sort -z |
xargs -0 sha256sum > sub-100206_before_sha256.txt
```

Since the destination currently contains relatively few files for one
subject, checksum verification should be cheap.

Also record counts:

``` bash
find "$SRC" -type f | wc -l
find "$DST" -type f | wc -l
```

Then run the file-level move and time it.

Afterward:

1.  Recompute checksums of all files that existed before.
2.  Confirm all original destination files are byte-identical.
3.  Count destination files.
4.  Inspect source leftovers.
5.  Confirm every leftover source file has a corresponding destination
    path.
6.  Confirm no source file is missing from both source and destination.
7.  Ideally compare the union against the tar/source manifest.

The crucial proof is:

> Every destination file that existed before the merge still exists and
> has the exact same SHA-256 hash afterward.

------------------------------------------------------------------------

## 14. Potential optimization for the full dataset

Do not necessarily perform 582k individual `mv` calls if there is a
cleaner safe optimization.

The coding agent should inspect the hierarchy.

### A. Entire top-level subject absent at destination

If:

``` text
SRC_ROOT/sub-XXXXXX exists
DST_ROOT/sub-XXXXXX does NOT exist
```

and both are on the same filesystem:

``` bash
mv SRC_ROOT/sub-XXXXXX DST_ROOT/sub-XXXXXX
```

is safe and extremely efficient because there is no collision.

### B. Subject exists at destination

For overlapping subjects such as `sub-100206`, descend into the tree.

If an entire source subdirectory is absent at the corresponding
destination, it may be safe to rename the entire directory at once.

Example:

``` text
SRC/sub-X/anat/
DST/sub-X/anat/ does not exist
```

could potentially be:

``` bash
mv SRC/sub-X/anat DST/sub-X/anat
```

But if `DST/sub-X/anat` already exists---even if empty---do not blindly
move the source directory over it. Descend further or merge its
children.

### C. Existing directories

Within aligned directories such as:

``` text
sub-X/func/
```

move only files whose exact destination path does not exist.

This recursive approach could greatly reduce process spawning and
metadata operations while preserving strict no-clobber semantics.

------------------------------------------------------------------------

## 15. Important implementation consideration

A naive shell loop invoking `mkdir` and `mv` for \~570k files may itself
be slow because it launches huge numbers of subprocesses.

The coding agent should consider implementing the full merge in **Python
using filesystem rename operations**, or another filesystem-native
approach.

However:

### Do NOT use `os.replace()`

because it overwrites existing destination paths.

If using Python, basic desired logic is approximately:

``` python
if not os.path.lexists(dst):
    os.rename(src, dst)
```

but there is technically a TOCTOU race between the existence check and
rename.

Since this dataset is not expected to be concurrently modified, that may
be acceptable operationally, but the agent should investigate whether
Linux `renameat2(..., RENAME_NOREPLACE)` is available. That provides
atomic:

> rename only if destination does not exist

semantics.

On Linux, `renameat2` with:

``` text
RENAME_NOREPLACE
```

would be ideal for strict no-clobber same-filesystem moves.

The agent can potentially invoke it through `ctypes` if Python's
standard library does not expose it.

This would provide stronger safety than:

``` python
if not exists:
    os.rename()
```

or relying on shell `mv -n`.

------------------------------------------------------------------------

## 16. Slurm requirements for the eventual full merge

The full merge should run as a CPU Slurm batch job, not on the login
node.

Match the established Torch header style:

``` bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=...
#SBATCH --mem=...
#SBATCH --job-name=hcp1200_merge
#SBATCH --output=/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200_transfer/merge_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200_transfer/merge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=asr655@nyu.edu

set -euo pipefail

module purge
```

No GPU should be requested.

The merge is filesystem/metadata work.

------------------------------------------------------------------------

## 17. Logging requirements for full merge

The eventual script/program should keep useful audit statistics,
ideally:

``` text
start timestamp
hostname
source root
destination root
source/destination filesystem device IDs

files examined
files moved
files skipped because destination existed
directories moved wholesale
directories created
bytes represented by moved files
errors
end timestamp
elapsed time
```

For skipped collisions, logging the relative paths to a file would be
useful:

``` text
existing_paths_skipped.txt
```

Avoid dumping hundreds of thousands of successful paths to Slurm stdout
unless necessary.

------------------------------------------------------------------------

## 18. Verification after full merge

Do **not** delete:

``` text
xcpd-0-9-1.tar
```

until verification is complete.

At completion, desired invariants include:

### Destination completeness

Every original source path from the tar/extracted dataset exists at the
final destination.

### Existing destination preservation

Every file that existed before the merge remains byte-for-byte
unchanged.

### No destructive operations

No destination file was deleted or replaced.

### Staging leftovers

Any files remaining in extracted staging should correspond to deliberate
collisions with pre-existing destination files.

Because the tar remains untouched, it can be used to reconstruct/inspect
the original source if needed.

Only after verification should cleanup of:

``` text
HCP1200_transfer/xcpd-0-9-1/
```

and eventually:

``` text
HCP1200_transfer/xcpd-0-9-1.tar
```

be considered.

------------------------------------------------------------------------

## 19. Current exact state / next action

**Nothing from the extracted staging tree has been merged yet.**

Transfer and extraction are complete and verified.

The immediate task is:

> Develop and run a meticulous **`sub-100206` smoke test** that moves
> only source files whose destination path does not already exist,
> preferably using same-filesystem atomic rename/no-clobber semantics.

Before the smoke test:

1.  Verify `SRC` and `DST` have the same filesystem device ID.
2.  Manifest/checksum the existing `sub-100206` destination files.
3.  Record source/destination counts.
4.  Run the no-clobber move and time it.
5.  Verify existing destination checksums are unchanged.
6.  Inspect staging leftovers.
7.  Confirm the resulting destination represents the union of the two
    trees.

If that succeeds, generalize the tested mechanism into a Slurm batch job
for the entire `xcpd-0-9-1` tree, optimizing by renaming whole
noncolliding directories where safely possible while descending into
directories that already exist.

**Safety is more important than maximum speed, but because source and
destination are on the same `/scratch` filesystem and \~406 GiB needs to
be incorporated, avoid physically copying hundreds of GB when safe
filesystem renames can accomplish the merge.**
