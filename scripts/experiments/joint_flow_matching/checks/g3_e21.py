"""Spec v4 E2.1 gate G3 (seed-0 pilot): reads the pilot records and predictions; exit 1 on failure (blocks the
dependent seed array).
  (a) corrector val mean decoded demeaned r >= ridge_cond's - 0.005;
  (b) flow sample means correlate >= 0.9 with the corrector's predictions (val, all conditions);
  (c) every metric finite; (d) the selected trial of every searched model stopped before the epoch cap.

    python scripts/experiments/joint_flow_matching/checks/g3_e21.py [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e21_run import OUT, PRED, TRAIN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    s = ap.parse_args().seed
    rec = {m: json.loads((OUT / f"{m}_seed{s}.json").read_text())
           for m in ("refs", "corrector", "corrector_id", "bridge", "flow", "latent2sc")}
    ok = True

    def mean_val(rows, model):
        return np.mean([r["demeaned_pearson"] for r in rows if r["model"] == model and r["split"] == "val"
                        and r.get("steps", 16) == 16])
    c, r = mean_val(rec["corrector"]["rows"], "corrector"), mean_val(rec["refs"]["rows"], "ridge_cond")
    a = c >= r - 0.005
    print(f"{'OK  ' if a else 'FAIL'} G3a corrector val dr {c:.4f} vs ridge {r:.4f} (>= ridge - 0.005)")
    pc, pf = np.load(PRED / f"corrector_seed{s}.npz")["val"], np.load(PRED / f"flow_seed{s}.npz")["val"]
    corr = float(np.corrcoef(pc.ravel(), pf.ravel())[0, 1])
    b = corr >= 0.9
    print(f"{'OK  ' if b else 'FAIL'} G3b flow sample means vs corrector predictions: r {corr:.3f} (>= 0.9)")
    vals = [v for m in rec.values() for row in m["rows"] for k, v in row.items() if isinstance(v, float)]
    vals += [v for m in rec.values() for key in ("dist", "dist_shuffled") for d in m.get(key, {}).values()
             for v in d.values() if isinstance(v, float)]
    cfin = bool(np.all(np.isfinite(vals)))
    print(f"{'OK  ' if cfin else 'FAIL'} G3c all {len(vals)} metrics finite")
    capped = [m for m in ("corrector", "corrector_id", "bridge", "flow")
              if [t for t in rec[m]["trials"] if t["hp"] == rec[m]["hp"]][0]["epochs_run"] >= TRAIN["max_epochs"]]
    d = not capped
    print(f"{'OK  ' if d else 'FAIL'} G3d selected trials stopped before the epoch cap {capped or ''}")
    for m in ("corrector", "corrector_id", "bridge", "flow", "latent2sc"):
        print(f"     {m}: {rec[m]['seconds_total'] / 60:.1f} min")
    ok = a and b and cfin and d
    print("ALL_OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
