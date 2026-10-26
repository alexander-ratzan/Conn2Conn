"""Regional BOLD timeseries cache (`Conn2Conn_data/timeseries/`) and windowed functional connectivity.

Cache layout (built by data/data_caching/build_ts_cache.py); one folder per condition x parcellation, named as in fc/:

    timeseries/parc-{P}_hemi-both[_task-{T}]/
        timeseries.npy     (T_total, P) float32: every subject's runs, concatenated in subject order
        subject_ids.npy    (N,) int64, sorted
        offsets.npy        (N + 1,) int64: subject i is rows offsets[i]:offsets[i + 1]
        runs.tsv           subject, run (LR1 / RL1 / LR2 / RL2), start (row in timeseries.npy), n_tr
        manifest.json      provenance, node labels, validation, sha256

A subject's rows are xcp-d's combined-run timeseries: its runs back to back, in the order of the xcp-d JSON sidecar
(rest LR1 RL1 LR2 RL2; tasks LR1 RL1). Values are xcp-d 0.9.1 denoised, band-pass filtered (~0.01-0.08 Hz) parcel
means, not z-scored; TR = 0.72 s. Windows never span a run boundary.

The fc/ matrices are a plain Pearson correlation of exactly these rows: corr(subject rows) reproduces every fc/
combined-run matrix to float32 rounding (max |diff| ~3e-8, all subjects / conditions / parcellations; per-run ~2e-7).
Any processing known for the FC (denoising, band-pass, no censoring, run concatenation) is therefore that of these
timeseries (Conn2Conn_data/timeseries/README.md).

    ts = TimeseriesCache("wm", "Glasser")
    x = ts.subject(100206)                        # (810, 360) float32
    starts = window_starts(ts.run_bounds(100206), length=144, step=24)
    C = window_fc(x, starts, 144)                 # (W, 360, 360)
"""
import json
import os

import numpy as np
import pandas as pd

DEFAULT_ROOT = "/scratch/asr655/neuroinformatics/Conn2Conn_data/timeseries"
TR = 0.72
CONDITIONS = ("rest", "emotion", "gambling", "language", "motor", "relational", "social", "wm")
PARCELLATIONS = ("Glasser", "4S456Parcels")


def cache_dir_name(condition, parcellation):
    """Same folder names as fc/ (rest has no task suffix)."""
    suffix = "" if condition == "rest" else f"_task-{condition}"
    return f"parc-{parcellation}_hemi-both{suffix}"


class TimeseriesCache:
    """Read access to one condition x parcellation folder; the timeseries array is memory-mapped by default."""

    def __init__(self, condition, parcellation="Glasser", root=DEFAULT_ROOT, mmap=True):
        self.condition, self.parcellation = condition, parcellation
        self.dir = os.path.join(root, cache_dir_name(condition, parcellation))
        self.data = np.load(os.path.join(self.dir, "timeseries.npy"), mmap_mode="r" if mmap else None)
        self.subject_ids = np.load(os.path.join(self.dir, "subject_ids.npy"))
        self.offsets = np.load(os.path.join(self.dir, "offsets.npy"))
        self.runs = pd.read_csv(os.path.join(self.dir, "runs.tsv"), sep="\t")
        with open(os.path.join(self.dir, "manifest.json")) as fh:
            self.manifest = json.load(fh)
        self.labels = self.manifest["node_labels"]
        self.tr = float(self.manifest.get("tr", TR))
        self._row = {int(s): i for i, s in enumerate(self.subject_ids)}
        self._runs_of = {int(s): g for s, g in self.runs.groupby("subject")}

    def __len__(self):
        return len(self.subject_ids)

    def __contains__(self, sid):
        return int(sid) in self._row

    def subject(self, sid):
        """(n_tr, P) float32 array (a copy, not a view of the memmap)."""
        i = self._row[int(sid)]
        return np.array(self.data[self.offsets[i]:self.offsets[i + 1]])

    def run_table(self, sid):
        """DataFrame (run, start, n_tr), start relative to the subject's first row."""
        g = self._runs_of[int(sid)]
        return pd.DataFrame({"run": g["run"].to_numpy(), "start": g["start"].to_numpy() - self.offsets[self._row[int(sid)]],
                             "n_tr": g["n_tr"].to_numpy()})

    def run_bounds(self, sid):
        """(n_runs, 2) int64 [start, stop) rows, relative to the subject's first row."""
        t = self.run_table(sid)
        return np.stack([t["start"].to_numpy(), (t["start"] + t["n_tr"]).to_numpy()], axis=1).astype(np.int64)

    def run(self, sid, run):
        t = self.run_table(sid).set_index("run")
        a, n = int(t.loc[run, "start"]), int(t.loc[run, "n_tr"])
        return self.subject(sid)[a:a + n]


def window_starts(bounds, length, step):
    """Start rows of windows of `length` rows taken every `step` rows inside each [start, stop) run."""
    out = [np.arange(a, b - length + 1, step) for a, b in np.asarray(bounds)]
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)


def corr(x):
    """Pearson correlation between the columns of x (T, P), float64. Constant columns give NaN rows / columns."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = x / np.sqrt(np.square(x).sum(0))
    return x.T @ x


def shrink(C, lam):
    """Linear shrinkage toward the identity, (1 - lam) C + lam I: keeps a correlation matrix's unit diagonal and makes
    a rank-deficient window matrix (length < P) positive definite for lam > 0."""
    P = C.shape[-1]
    return (1.0 - lam) * C + lam * np.eye(P)


def window_fc(x, starts, length, shrinkage=0.0, upper=False, batch=64, dtype=np.float32):
    """Pearson FC of each window x[s:s + length] for s in `starts`.

    Returns (W, P, P), or (W, P(P-1)/2) upper triangles (k=1) with `upper=True`; computed in float64, in batches of
    `batch` windows, cast to `dtype`.
    """
    x = np.asarray(x)
    P = x.shape[1]
    starts = np.asarray(starts, dtype=np.int64)
    iu = np.triu_indices(P, k=1)
    out = np.empty((len(starts), len(iu[0])) if upper else (len(starts), P, P), dtype=dtype)
    offs = np.arange(length)
    for b in range(0, len(starts), batch):
        X = x[starts[b:b + batch, None] + offs].astype(np.float64)       # (w, L, P)
        X -= X.mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            X /= np.sqrt(np.square(X).sum(1, keepdims=True))
        C = np.matmul(X.transpose(0, 2, 1), X)
        if shrinkage:
            C = shrink(C, shrinkage)
        out[b:b + batch] = C[:, iu[0], iu[1]] if upper else C
    return out
