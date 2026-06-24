#!/usr/bin/env bash
# Build the findings-ledger PDF: regenerate F1-F10 figures, then compile LaTeX.
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/user/dev-env/bin/python
export PATH="/Users/user/Library/TinyTeX/bin/universal-darwin:$PATH"
echo "[1/2] regenerating F1-F13 figures ..."
"$PY" make_findings.py
"$PY" make_findings_extra.py
echo "[2/3] compiling findings document ..."
latexmk -pdf -interaction=nonstopmode conn2conn-findings.tex

echo "[3/3] building pick-and-choose slideshow ..."
"$PY" make_slides.py
"$PY" build_slides.py
latexmk -pdf -interaction=nonstopmode slides.tex

echo "Done -> conn2conn-findings.pdf , slides.pdf"
