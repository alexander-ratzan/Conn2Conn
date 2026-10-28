"""windowed_fc_gain G7a: closed-form SC -> latent models trained on part-sample arms (one seed, both routes, all arms).

Per route (cca, pca) and arm (g7_common.ARMS): ridge per condition (point; decoded-edge scores as E2.1) and the
linear-Gaussian (ridge means + one covariance: `own` = all units' residuals, `fullcov` = full-target residuals only),
scored against the full val / test targets with E2.1's distribution metrics. Also runs the G7.0 checks and caches each
arm's ridge means / covariances for the flows (run_g7b.py).

    python scripts/experiments/windowed_fc_gain/run_g7a.py --seed 0      # -> results/g7/closed_seed0.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g7_common import ARMS, FC, G7_CACHE, ROUTES, Route, g7_checks, load_parts, units  # noqa: E402
from common import score  # noqa: E402
from e21_run import K_SAMPLES, distribution_metrics  # noqa: E402

OUT = HERE / "results" / "g7"


def gauss_samples(means, cov, K=K_SAMPLES, seed=0):
    """E2.1's linear-Gaussian cloud: each sample picks a condition's ridge mean and adds shared Gaussian noise."""
    rng = np.random.default_rng(seed)
    n, d = means.shape[1], means.shape[2]
    pick = rng.integers(0, 8, size=(K, n))
    return means[pick, np.arange(n)[None, :]] + rng.multivariate_normal(np.zeros(d), cov, size=(K, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    seed = ap.parse_args().seed
    t0 = time.time()
    rec = dict(seed=seed, checks={}, units={}, ks={}, rows=[], dist=[])
    G7_CACHE.mkdir(parents=True, exist_ok=True)
    for route in ROUTES:
        R = Route(seed, route)
        full, parts, ok = load_parts(seed, route)
        rec["checks"][route] = g7_checks(seed, route, R, full)
        print(f"seed {seed} {route}: G7.0 {rec['checks'][route]}", flush=True)
        tr = R.L.sd.splits["train"]
        parts_tr, ok_tr = {lv: P[:, :, tr] for lv, P in parts.items()}, ok[tr]
        for arm in ARMS:
            subj, typ, tgt = units(arm, R.enc["train"], parts_tr, ok_tr)
            rc, ks, cov = R.fit_arm(subj, typ, tgt)
            np.savez(G7_CACHE / f"closed_seed{seed}_{route}_{arm}.npz", rc_train=rc["train"], rc_val=rc["val"],
                     rc_test=rc["test"], subj=subj, typ=typ, tgt=tgt, ks=np.array(ks),
                     **{f"cov_{k}": v for k, v in cov.items()})
            rec["units"][f"{route}/{arm}"] = dict(n_units=int(len(subj)), n_subjects=int(len(np.unique(subj))),
                                                  by_type=np.bincount(typ, minlength=3).tolist())
            rec["ks"][f"{route}/{arm}"] = ks
            for split in ("val", "test"):
                for m, v in enumerate(FC):
                    sc = score(R.L.decode(m, rc[split][m]), R.L.sd.edges[(split, v)], R.L.sd.train_mean[v])
                    rec["rows"].append(dict(model="ridge", route=route, arm=arm, condition=v, split=split, **sc))
                for name, C in cov.items():
                    dm = distribution_metrics(gauss_samples(rc[split], C), R.enc[split])
                    rec["dist"].append(dict(model=f"gauss_{name}", route=route, arm=arm, split=split,
                                            **{k: x for k, x in dm.items() if k not in ("per_subject", "example")}))
            print(f"seed {seed} {route} {arm}: units {len(subj)}, ridge k {ks} ({time.time() - t0:.0f} s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"closed_seed{seed}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"wrote {OUT / f'closed_seed{seed}.json'} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
