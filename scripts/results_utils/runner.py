"""
runner.py
=========
Shared plumbing for config-driven results experiments (`scripts/experiments/<slug>/run.py`):
CLI, config loading, the tracked records-cache contract, table/figure writing, and the manifest.

Each experiment's run.py keeps only what is specific to it: how to scrape, which tables to
build, and how figure specs map to data.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from .plots import figure_name, render_figure, save_figure
from .records import REPO_ROOT, load_records_cache


def rel(path) -> str:
    """Repo-relative path for messages/manifest; absolute when outside the repo."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def arg_parser(doc: str, here: Path) -> argparse.ArgumentParser:
    """Standard CLI: --config, --cache (tracked records.json), --rescrape, --out."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=here / "config.yml")
    ap.add_argument("--cache", type=Path, default=here / "records.json",
                    help="tracked records snapshot (default: records.json next to run.py)")
    ap.add_argument("--rescrape", action="store_true", help="refresh the records cache from W&B first")
    ap.add_argument("--out", type=Path, default=None, help="override config output_dir (default: the experiment folder)")
    return ap


def load_config(path: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("selection", {"metric": "val_demeaned_r", "mode": "max"})
    cfg.setdefault("figures", [])
    return cfg


def load_cached(cache_path: Path, want: dict, exact: tuple = ()) -> tuple[list, dict]:
    """
    Load the tracked records cache, exiting with a clear message if it is missing or does not
    cover `want` (list values must be subsets of the cache's; keys in `exact` must match exactly).
    """
    if not cache_path.exists():
        sys.exit(f"No records cache at {cache_path}. Run with --rescrape to build it from W&B.")
    records, meta = load_records_cache(cache_path)
    problems = []
    for key, value in want.items():
        have = meta.get(key)
        if key in exact or not isinstance(value, list):
            if have != value:
                problems.append(f"{key}: cache={have!r} config={value!r}")
        else:
            missing = [x for x in value if x not in (have or [])]
            if missing:
                problems.append(f"{key} not in cache: {missing}")
    if problems:
        sys.exit("Cache does not match config; run with --rescrape.\n  " + "\n  ".join(problems))
    print(f"Loaded {len(records)} records from {rel(cache_path)} (scraped {meta.get('scraped_at', 'unknown')})")
    return records, meta


def scrape_metadata(want: dict) -> dict:
    from .records import WANDB_ENTITY, WANDB_PROJECT

    meta = dict(want)
    meta["scraped_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["wandb"] = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    return meta


def markdown_table(df) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_table(df, out_dir: Path, name: str, index: bool = False, markdown: bool = False) -> list:
    """Write <name>.csv (and <name>.md when `markdown`) into out_dir; returns written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / f"{name}.csv"]
    df.to_csv(written[0], index=index)
    if markdown:
        written.append(out_dir / f"{name}.md")
        written[-1].write_text(markdown_table(df))
    return written


def write_figures(cfg: dict, out_dir: Path, resolve: Callable[[dict], tuple[dict, dict]]) -> list:
    """
    Render every entry of cfg["figures"]. `resolve(spec)` returns (spec_for_render_figure, data)
    where `data` holds the arguments the experiment supplies (records, seed_df, row_specs, ...).
    """
    import matplotlib.pyplot as plt

    formats = tuple(cfg.get("formats", ["png"]))
    dpi = int(cfg.get("dpi", 300))
    names = [figure_name(spec) for spec in cfg["figures"]]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        sys.exit(f"Figure specs produce duplicate file names {dupes}; give them explicit `name`s.")
    written = []
    for spec, name in zip(cfg["figures"], names):
        render_spec, data = resolve(spec)
        fig, _, _ = render_figure(render_spec, **data)
        written += save_figure(fig, out_dir / name, formats=formats, dpi=dpi)
        plt.close(fig)
        print(f"  figure: {name}")
    return written


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(out_dir: Path, config_path: Path, cache_path: Path, written: list,
                   cache_meta: Optional[dict] = None) -> Path:
    if cache_meta is None:
        _, cache_meta = load_records_cache(cache_path)
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": rel(config_path),
        "records_cache": rel(cache_path),
        "cache_metadata": cache_meta,
        "outputs": [str(Path(p).resolve().relative_to(out_dir)) for p in written],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(written)} files to {rel(out_dir)}/")
    return path
