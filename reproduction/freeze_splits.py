#!/usr/bin/env python3
"""Freeze the 10 seed splits ONCE -> reproduction/splits/seed{0..9}.json (BP-2).

These JSON files are the SOURCE OF TRUTH for which subjects land in train/val/test for each
seed. Every grid runner loads its data via _setup and HARD-ASSERTS the produced split matches
these files (load_split_checked) — so the split can never silently drift or misalign.

Subject availability is identical across parcellations (verified: SC/FC symdiff 0), so the
splits are parcellation-independent. We still CROSS-CHECK each seed against BOTH parcellations
here and record the verification, so "identical across parcellations" is proven per-seed, not
assumed.

Writes (committed back to the repo so they are definite/versioned):
  splits/seed{N}.json        {seed, n_*, train_ids, val_ids, test_ids, parc_verified, git_commit}
  splits/SPLITS_MANIFEST.json
"""
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import (  # noqa: E402
    load_seed_split, set_parcellation, subject_ids_for, split_json_path,
    SPLITS_DIR, PARCELLATIONS, git_commit,
)

N_SEEDS = 10


def ids_for_seed(seed: int, parc: str):
    set_parcellation(parc)
    sp = load_seed_split(seed=seed)
    base = sp["base"]
    part = base.trainvaltest_partition_indices
    return (subject_ids_for(base, part["train"]),
            subject_ids_for(base, part["val"]),
            subject_ids_for(base, part["test"]))


def main():
    commit = git_commit()
    manifest = {"git_commit": commit, "n_seeds": N_SEEDS,
                "parcellations_verified": PARCELLATIONS, "seeds": []}
    for seed in range(N_SEEDS):
        tr, va, te = ids_for_seed(seed, PARCELLATIONS[0])  # Glasser
        # cross-check every other parcellation gives the IDENTICAL ordered split
        parc_verified = {PARCELLATIONS[0]: True}
        for parc in PARCELLATIONS[1:]:
            tr2, va2, te2 = ids_for_seed(seed, parc)
            ok = (tr2 == tr and va2 == va and te2 == te)
            parc_verified[parc] = bool(ok)
            if not ok:
                raise AssertionError(
                    f"seed{seed}: {parc} split differs from {PARCELLATIONS[0]} "
                    f"(set-equal train={set(tr2)==set(tr)} val={set(va2)==set(va)} "
                    f"test={set(te2)==set(te)}) — parcellation-independence VIOLATED")
        rec = {"seed": seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "n_total": len(tr) + len(va) + len(te),
               "train_ids": tr, "val_ids": va, "test_ids": te,
               "parc_verified": parc_verified, "git_commit": commit}
        split_json_path(seed).write_text(json.dumps(rec))
        manifest["seeds"].append({k: rec[k] for k in
                                  ("seed", "n_train", "n_val", "n_test", "n_total", "parc_verified")})
        print(f"[freeze] seed{seed}: train={len(tr)} val={len(va)} test={len(te)} "
              f"total={rec['n_total']} parc_verified={parc_verified}", flush=True)

    (SPLITS_DIR / "SPLITS_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[freeze] wrote {N_SEEDS} frozen splits + manifest -> {SPLITS_DIR}", flush=True)


if __name__ == "__main__":
    main()
