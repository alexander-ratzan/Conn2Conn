"""Shareable (claude.ai artifact) build of the timeseries viewer: same page, data split into one file per subject.

The local viewer embeds every subject in one ~40 MB page; an artifact page is limited to 16 MB, but files published
next to it can be fetched by relative URL. This build writes, from the local viewer's data cache (no recomputation):
    <out>/timeseries_viewer.html        page content for the artifact publisher (no <html>/<head>/<body>: the publisher
                                        wraps it): kit CSS + page CSS, the metadata block, kit.js and the page script
    <out>/data/sub-<id>.json            {"<id>/<condition>": set} for that subject's 8 conditions (~10 MB each)
The page fetches a subject's file when it is first selected (the first subject at load). More subjects = more files:
rebuild the local viewer with --subjects ..., then rerun this script and republish.

    python scripts/data_viewers/timeseries_viewer/build_timeseries_artifact.py --out <dir> [--parcellation Glasser]
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from scripts.data_viewers.viewer_kit import Viewer  # noqa: E402
from scripts.data_viewers.viewer_kit.viewer import KIT, NOTE, TOP, _block, _dumps  # noqa: E402

LOADER = r'''
// ---------- per-subject data files (artifact build: data/sub-<id>.json, fetched on first use) ----------
const LOADED = {};
function loadSubject(sid) {
  if (!LOADED[sid]) {
    $("loadStatus").textContent = `Loading sub-${sid} …`;
    $("subject").disabled = true;
    LOADED[sid] = fetch(`data/sub-${sid}.json`)
      .then(r => { if (!r.ok) throw new Error(`could not load data/sub-${sid}.json (HTTP ${r.status})`); return r.json(); })
      .then(sets => { Object.assign(D.sets, sets); })
      .catch(e => { delete LOADED[sid]; Kit.fail(e.message); throw e; })
      .finally(() => { $("loadStatus").textContent = ""; $("subject").disabled = false; });
  }
  return LOADED[sid];
}
'''
EXTRA_CSS = """
/* artifact build: sticky controls clear the phone's top inset; controls inert until the first subject has loaded */
.controls { top: env(safe-area-inset-top, 0px); }
.controls[inert] { opacity: 0.6; }
#loadStatus { margin-left: 8px; color: var(--text-muted); }
"""
DATA_NOTE = ("Human Connectome Project (WU-Minn) Open Access data, shared with collaborators who have accepted the HCP "
             "Data Use Terms; not for redistribution outside those terms.")


def replace_once(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of: {old[:70]!r}")
    return s.replace(old, new)


def joined(v):
    """viewer_kit's chunked strings joined back into plain strings (the page fetches whole JSON, no line limits)."""
    if isinstance(v, dict):
        if "__chunks__" in v:
            return "".join(v["__chunks__"])
        return {k: joined(x) for k, x in v.items()}
    if isinstance(v, list):
        return [joined(x) for x in v]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--parcellation", default="Glasser")
    a = ap.parse_args()
    viewer = Viewer(HERE / "build_timeseries_viewer.py")
    meta, sets = viewer.load_cache(a.parcellation)

    page = viewer.template.read_text()
    head = re.search(r"<head>(.*)</head>", page, re.S).group(1)
    body = re.search(r"<body>(.*)</body>", page, re.S).group(1)
    title = re.search(r"<title>.*?</title>", head).group(0)
    page_css = re.search(r"<style>(.*?)</style>", head, re.S).group(1)
    body = NOTE.sub("", body)
    body = replace_once(body, "<!--KIT:TOP-->", TOP)
    body = replace_once(body, '<div class="sub" id="subtitle"></div>',
                        '<div class="sub"><span id="subtitle"></span><span id="loadStatus" role="status"></span></div>')
    body = replace_once(body, '<div class="controls">', '<div class="controls" id="controls" inert>')
    body = replace_once(body, "<footer>\n", f'<footer>\n  <div class="note">{DATA_NOTE}</div>\n')
    body = replace_once(body, "const D = Kit.load();", "const D = Kit.load();" + LOADER)
    body = replace_once(body, '$("subject").onchange = e => { S.subject = e.target.value; updateStarts(); drawAll(); };',
                        '$("subject").onchange = e => { const sid = e.target.value;\n'
                        '  loadSubject(sid).then(() => { S.subject = sid; updateStarts(); drawAll(); }, () => { $("subject").value = S.subject; }); };')
    body = replace_once(body, "LUT = Kit.divergingLUT(); drawColorbar(); updateStarts(); drawAll(); selfCheck();",
                        'LUT = Kit.divergingLUT(); drawColorbar();\n'
                        'loadSubject(S.subject).then(() => { $("controls").inert = false; updateStarts(); drawAll(); selfCheck(); });')

    body = replace_once(body, "<!--KIT:DATA-->", _block('id="kit-meta"', _dumps(meta)) +
                        f"\n<script>\n{(KIT / 'kit.js').read_text()}</script>")

    out = a.out
    (out / "data").mkdir(parents=True, exist_ok=True)
    html = (f"{title}\n<style>\n{(KIT / 'kit.css').read_text()}{EXTRA_CSS}</style>\n<style>{page_css}</style>\n"
            f"{body.strip()}\n")
    (out / "timeseries_viewer.html").write_text(html)
    by_subject = {}
    for key, s in sets.items():
        by_subject.setdefault(key.split("/")[0], {})[key] = joined(s)
    for sid, d in by_subject.items():
        p = out / "data" / f"sub-{sid}.json"
        p.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB, {len(d)} conditions)")
    size = (out / "timeseries_viewer.html").stat().st_size
    print(f"wrote {out / 'timeseries_viewer.html'} ({size / 1e3:.0f} kB); subjects {sorted(by_subject)}")
    if size > 16e6 or any((out / "data" / f"sub-{s}.json").stat().st_size > 16e6 for s in by_subject):
        raise SystemExit("a file exceeds the 16 MB artifact limit")


if __name__ == "__main__":
    main()
