"""
conftest.py — data_source unit tests

The common/data_source/__init__.py imports every connector, which pulls in
heavy optional dependencies (googleapiclient, slack_sdk, jira, etc.) that are
not installed in a minimal test environment.

This file pre-registers a lightweight common.data_source package stub in
sys.modules BEFORE pytest collects any test modules.  The stub prevents
__init__.py from running, so only the modules that sanad_connector.py actually
needs are imported (config, models, exceptions, interfaces).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_SOURCE = _REPO_ROOT / "common" / "data_source"


def _load_direct(dotted_name: str, file_path: Path) -> ModuleType:
    """Load a single .py file into sys.modules under a given dotted name."""
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 1. Ensure parent packages exist as lightweight stubs ─────────────────────
for _dotted, _path in [
    ("common", _REPO_ROOT / "common"),
    ("common.data_source", _DATA_SOURCE),
]:
    if _dotted not in sys.modules:
        _stub = ModuleType(_dotted)
        _stub.__path__ = [str(_path)]
        _stub.__package__ = _dotted
        sys.modules[_dotted] = _stub

# Attach data_source onto the common stub so attribute access works
sys.modules["common"].data_source = sys.modules["common.data_source"]  # type: ignore[attr-defined]

# ── 2. Load only the files that sanad_connector.py imports, in order ─────────
#  (This avoids any heavy transitive deps from the full __init__.py)
_load_direct("common.data_source.config", _DATA_SOURCE / "config.py")
_load_direct("common.data_source.models", _DATA_SOURCE / "models.py")
_load_direct("common.data_source.exceptions", _DATA_SOURCE / "exceptions.py")
_load_direct("common.data_source.interfaces", _DATA_SOURCE / "interfaces.py")
_load_direct("common.data_source.sanad_connector", _DATA_SOURCE / "sanad_connector.py")

# Expose them as attributes on the package stub so normal imports resolve
_pkg = sys.modules["common.data_source"]
for _attr in ("config", "models", "exceptions", "interfaces", "sanad_connector"):
    setattr(_pkg, _attr, sys.modules[f"common.data_source.{_attr}"])
