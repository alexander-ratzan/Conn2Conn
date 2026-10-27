"""viewer_kit: the shared base of Conn2Conn's self-contained HTML viewers.

A viewer is one folder holding its builder, its template and the page it builds (the page is generated, not tracked):

    scripts/data_viewers/<name>/                    data viewers (any dataset, not one experiment's results)
    scripts/experiments/<slug>/viewers/<name>/      an experiment's viewers
        build_<name>.py        collects the data, then Viewer(__file__).write(meta, sets)
        <name>.template        the page: HTML + page CSS + page JS, with the kit markers below
        <name>[_<variant>].html     built page (gitignored), open this one
        <name>[_<variant>]_data.json  optional data cache for --from-cache rebuilds (gitignored)

Template contract (checked at build time):
    <!--KIT:HEAD-->     in <head>, before the page's own <style>: kit.css
    <!--KIT:TOP-->      first thing in <body>: the red error line (#kit-err) and the tooltip (#kit-tip)
    <!--KIT:DATA-->     before the page's <script>: the metadata block, one block per data set, then kit.js
    one line holding class="kit-template-note": visible only when the template itself is opened (removed at build)
The page script starts with `const D = Kit.load();` -> {...meta, sets: {key: set}}. An optional <div id="kit-checks">
shows Kit.check(label, ok, detail) results (page JS vs Python reference values).

Data: `meta` is a small dict; `sets` maps a key (e.g. "100206/rest") to one dict per data set, each written to its own
<script type="application/json"> block. Long strings (base64 arrays) go in as packed(array) / chunked(str) and are split
into lines of at most CHUNK characters: editor previews (Cursor) fail on multi-megabyte lines. NaN / inf -> null.
"""
import base64
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
CHUNK = 200_000
NL_DEPTH = 3
MARKERS = ("<!--KIT:HEAD-->", "<!--KIT:TOP-->", "<!--KIT:DATA-->")
NOTE = re.compile(r'^[^\n]*class="kit-template-note"[^\n]*\n', re.M)
TOP = ('<div id="kit-err" role="alert" hidden></div><div id="kit-tip"></div>\n'
       '<noscript><p style="color:#d03b3b;padding:0 20px">This viewer needs JavaScript: open it in a browser or a preview '
       'that runs scripts (Cursor: "Live Preview: Show Preview").</p></noscript>')
SIF = "/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"   # holds node (v18) for the smoke test


# ---------- encoding helpers ----------
def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def chunked(s):
    """A long string, written one chunk per line; the page sees the joined string."""
    return {"__chunks__": [s[i:i + CHUNK] for i in range(0, len(s), CHUNK)]}


def packed(a):
    """An array's raw little-endian bytes as chunked base64 (decode with Kit.int16 / Kit.float32 / Kit.bytes)."""
    return chunked(b64(a))


def int16_columns(x):
    """int16 codes and float64 per-column scales with x ~= q * scale."""
    x = np.asarray(x, dtype=np.float64)
    scale = np.abs(x).max(0) / 32767.0
    scale[scale == 0] = 1.0
    return np.round(x / scale).astype("<i2"), scale


def int16_unit(x):
    """Values in [-1, 1] (correlations) as int16 codes, x ~= q / 32767."""
    return np.round(np.clip(np.asarray(x, dtype=np.float64), -1, 1) * 32767).astype("<i2")


def round_list(a, nd=None):
    """Flat list for JSON: NaN / inf -> None, optionally rounded."""
    return [None if not np.isfinite(v) else (float(v) if nd is None else round(float(v), nd))
            for v in np.asarray(a, dtype=float).ravel()]


# ---------- JSON writing ----------
def _scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return v


def _dumps(v, depth=0):
    """Compact JSON; chunked strings one chunk per line; dict fields one per line down to depth NL_DEPTH (no huge line)."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, dict):
        if "__chunks__" in v:
            return '{"__chunks__":[\n' + ",\n".join(json.dumps(c) for c in v["__chunks__"]) + "\n]}"
        sep = ",\n" if depth < NL_DEPTH else ","
        return "{" + sep.join(json.dumps(str(k)) + ":" + _dumps(x, depth + 1) for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_dumps(x, depth + 1) for x in v) + "]"
    return json.dumps(_scalar(v), allow_nan=False)


def _block(attrs, body):
    return f"<script {attrs} type=\"application/json\">\n{body.replace('</', '<' + chr(92) + '/')}\n</script>"


# ---------- the viewer ----------
class Viewer:
    """One viewer folder: Viewer(__file__) from its build_<name>.py (name = folder name unless given)."""

    def __init__(self, builder_file, name=None):
        self.dir = Path(builder_file).resolve().parent
        self.name = name or self.dir.name
        self.template = self.dir / f"{self.name}.template"

    def _stem(self, variant):
        return self.name + (f"_{variant}" if variant else "")

    def html_path(self, variant=None):
        return self.dir / f"{self._stem(variant)}.html"

    def cache_path(self, variant=None):
        return self.dir / f"{self._stem(variant)}_data.json"

    # cache: the exact meta + sets of the last build, so template / kit edits rebuild without recomputing
    def save_cache(self, meta, sets, variant=None):
        p = self.cache_path(variant)
        p.write_text(json.dumps({"meta": meta, "sets": sets}, separators=(",", ":"), allow_nan=False, default=_scalar))
        return p

    def load_cache(self, variant=None):
        p = self.cache_path(variant)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: build once without --from-cache")
        c = json.loads(p.read_text())
        return c["meta"], c["sets"]

    def render(self, meta, sets):
        page = self.template.read_text()
        for m in MARKERS:
            if page.count(m) != 1:
                raise ValueError(f"{self.template.name}: marker {m} must appear exactly once")
        if len(NOTE.findall(page)) != 1:
            raise ValueError(f'{self.template.name}: needs one line with class="kit-template-note" (shown if the '
                             "template itself is opened)")
        data = "\n".join([_block('id="kit-meta"', _dumps(meta))]
                         + [_block(f'class="kit-set" data-key="{k}"', _dumps(s)) for k, s in sets.items()]
                         + [f"<script>\n{(KIT / 'kit.js').read_text()}</script>"])
        page = NOTE.sub("", page)
        page = page.replace(MARKERS[0], f"<style>\n{(KIT / 'kit.css').read_text()}</style>")
        page = page.replace(MARKERS[1], TOP)
        head, tail = page.split(MARKERS[2])     # data last: its text is never searched for markers
        return head + data + tail

    def write(self, meta, sets, variant=None, cache=True, smoke=False):
        """Render, check line lengths, write <name>[_<variant>].html (and the data cache); optionally smoke-test."""
        html = self.render(meta, sets)
        longest = max(map(len, html.splitlines()))
        if longest > 2 * CHUNK:
            raise ValueError(f"a line of {longest} characters: pass long strings as packed() / chunked()")
        out = self.html_path(variant)
        out.write_text(html)
        if cache:
            self.save_cache(meta, sets, variant)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(sets)} data sets, longest line {longest})")
        if smoke:
            smoke_test(out)
        return out

    @staticmethod
    def add_args(ap):
        ap.add_argument("--from-cache", action="store_true", help="rebuild the page from the last build's data cache")
        ap.add_argument("--smoke", action="store_true", help="run the page's JS under node with a DOM stub after building")
        return ap


def smoke_test(path, timeout=600):
    """Run the built page's scripts under node (cuda image) with a DOM stub; raise if the page reports an error or a
    failed Kit.check. Layout and drawing are not tested (no browser on the cluster)."""
    cmd = ["node", str(KIT / "smoke_test.js"), str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        res = subprocess.run(["singularity", "exec", "--bind", "/scratch", SIF, *cmd], capture_output=True, text=True,
                             timeout=timeout)
    print(res.stdout.strip())
    if res.returncode != 0:
        raise RuntimeError(f"smoke test failed for {path}:\n{res.stderr.strip() or res.stdout.strip()}")
