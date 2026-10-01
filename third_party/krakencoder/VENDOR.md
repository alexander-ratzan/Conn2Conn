# Vendored: Krakencoder

Unmodified copy of upstream [kjamison/krakencoder](https://github.com/kjamison/krakencoder) (MIT, see `LICENSE`),
used by the retrainable `Krakencoder` benchmark model (`scripts/krakencoder/train_krakencoder.py`).

| | |
|---|---|
| Commit | `b57e39c2771c36ab39d2f24a5d4355f3d375624d` (2025-10-31, "add example FC, SC inputs. add ipywidgets to requirements") |
| Package version | `krakencoder 1.0.0` (`krakencoder/_version.py`) |
| Files kept | `krakencoder/` (package incl. `resources/`), `run_training.py`, `run_model.py`, `LICENSE`, `README.md`, `requirements.txt` |
| Files dropped | example data, images, notebooks, MATLAB code, tests, helper scripts |
| Vendored | 2026-10-01 |

Why this commit: it is byte-identical to the `krakencoder 1.0.0` installed in the `kraken_env` overlay, and it is the base
of `krakencoder_experimental/` (gitignored local copy that produced the March 2026 cached predictions; it adds a
demeaned-MSE loss, debug prints and `canonical_data_flavor(accept_unknowns=True)` on top of this commit).

**Do not edit files here.** Repo-side adaptations live in the wrapper (`scripts/krakencoder/_vendor_entry.py`): it puts
this directory first on `sys.path`, checks that `krakencoder` is imported from here, and makes
`canonical_data_flavor` accept flavor names that are not in upstream's `flavordb.json` (our `Glasser` /
`4S456Parcels` flavors), which is the one setting the local copy changed in code. To update, replace the files from a
new upstream commit and record it above.
