"""Spec v4 E4.0 reference: identity in code space (B0-codes), every seed.

B0 copies the source's full edge deviation; the models only see and predict k block-PCA codes. B0-codes copies the
source's k codes into the target condition (decoded with the target's mean and scale; multi-source: mean of the
sources' codes), separating "identity keeps information outside the codes" from "the regressions shrink".

    python scripts/experiments/cross_condition_translation/run_identity_codes.py   # -> results/identity_codes.json
"""
import argparse
import json

import numpy as np

from e4_common import CFG, FC, HERE, SC_ONLY, Scorer, SeedData, cells, sources_of, suffix

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=CFG["k"])
K = ap.parse_args().k
rows = []
for seed in CFG["seeds"]:
    sd = SeedData(seed, "main")
    sc = Scorer(sd)
    for split in ("val", "test"):
        for src, tgt, with_sc in cells():
            if with_sc or src == SC_ONLY:
                continue
            z = np.mean([sd.code(a, split, K) for a in sources_of(src, tgt)], axis=0)
            rows.append(dict(seed=seed, model="B0-codes", source=src, target=tgt, with_sc=False, split=split,
                             **sc(tgt, split, z)))
    print(f"seed {seed} done", flush=True)
(HERE / "results" / f"identity_codes{suffix(K)}.json").write_text(json.dumps(dict(rows=rows), default=float))
print(f"wrote results/identity_codes{suffix(K)}.json ({len(rows)} rows)")
