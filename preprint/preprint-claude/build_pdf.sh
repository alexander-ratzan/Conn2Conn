#!/usr/bin/env bash
# Build the preprint PDF: regenerate all figures/tables, then compile LaTeX.
# Usage: bash build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=/Users/user/dev-env/bin/python
export PATH="/Users/user/Library/TinyTeX/bin/universal-darwin:$PATH"

echo "[1/2] regenerating figures + tables ..."
"$PY" make_all.py

echo "[2/2] compiling LaTeX (main + supplement) ..."
latexmk -pdf -interaction=nonstopmode conn2conn-preprint.tex
latexmk -pdf -interaction=nonstopmode conn2conn-supplement.tex

echo "Done -> conn2conn-preprint.pdf , conn2conn-supplement.pdf"
