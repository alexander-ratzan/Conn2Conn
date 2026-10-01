"""
Run a script from the vendored upstream Krakencoder (vendor/ next to this file) with the vendored package.

    python models/architectures/krakencoder/_vendor_entry.py run_training <run_training.py args...>
    python models/architectures/krakencoder/_vendor_entry.py run_model    <run_model.py args...>

The vendored files are never edited (vendor/VENDOR.md). This entry point:
  1. puts vendor/ first on sys.path and checks that `krakencoder` is imported from there
     (the kraken_env overlay also has an installed `krakencoder 1.0.0`; the vendored commit is identical to it,
     but the check keeps runs pinned to the tracked copy);
  2. makes `canonical_data_flavor` default to accept_unknowns=True, so flavor names that are not in upstream's
     flavordb.json (our Glasser / 4S456Parcels flavors) are accepted. This is the one code setting the old local
     copy (krakencoder_experimental/) changed; explicit accept_unknowns arguments are still honoured;
  3. runs the requested upstream script as __main__.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
SCRIPTS = {"run_training": VENDOR_DIR / "run_training.py", "run_model": VENDOR_DIR / "run_model.py"}


def _patch_flavor_lookup() -> None:
    import krakencoder.data as kdata

    original = kdata.canonical_data_flavor

    def canonical_data_flavor(conntype, only_if_brackets=False, return_groupname=False, accept_unknowns=True):
        return original(conntype, only_if_brackets=only_if_brackets, return_groupname=return_groupname,
                        accept_unknowns=accept_unknowns)

    canonical_data_flavor.__doc__ = original.__doc__
    # Rebind everywhere the name was star-imported (later star-imports pick up the patched module attribute).
    for name, module in list(sys.modules.items()):
        if (name == "krakencoder" or name.startswith("krakencoder.")) and \
                getattr(module, "canonical_data_flavor", None) is original:
            module.canonical_data_flavor = canonical_data_flavor


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        sys.exit(f"usage: {Path(__file__).name} {{{','.join(SCRIPTS)}}} [args...]")
    script = SCRIPTS[sys.argv[1]]
    sys.path.insert(0, str(VENDOR_DIR))
    import krakencoder

    loaded = Path(krakencoder.__file__).resolve()
    if VENDOR_DIR not in loaded.parents:
        sys.exit(f"krakencoder imported from {loaded}, expected the vendored copy under {VENDOR_DIR}")
    _patch_flavor_lookup()
    print(f"[krakencoder vendor] {krakencoder.__version__} from {loaded.parent}", flush=True)
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
