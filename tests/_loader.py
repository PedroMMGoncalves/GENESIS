"""Helper to import ``genesis_toolbox.pyt`` as a Python module.

The ``.pyt`` extension is not part of the default import machinery,
and the file lives at the repository root, so plain ``import`` will
not find it. ``importlib.util.spec_from_file_location`` reads the
file as Python source — the top-level ``import arcpy`` requires
arcpy to be importable, i.e. tests must run from ArcGIS Pro's Python
environment.
"""

import importlib.util
import os
import sys
from importlib.machinery import SourceFileLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLBOX_PATH = os.path.normpath(os.path.join(_HERE, "..", "genesis_toolbox.pyt"))


def load_toolbox():
    """Load ``genesis_toolbox.pyt`` and return the resulting module.

    The ``.pyt`` extension is not part of ``importlib``'s default
    suffix list, so ``spec_from_file_location`` returns ``None`` if
    we let it pick a loader on its own. We hand it an explicit
    ``SourceFileLoader``, which treats the file as plain Python
    source. Cached on first call via ``sys.modules``; a partial load
    (exception during ``exec_module``) is popped so the next attempt
    starts from a clean slate instead of returning a half-built
    module.
    """
    cached = sys.modules.get("genesis_toolbox")
    if cached is not None and hasattr(cached, "AsterMosaic"):
        return cached

    sys.modules.pop("genesis_toolbox", None)

    loader = SourceFileLoader("genesis_toolbox", _TOOLBOX_PATH)
    spec = importlib.util.spec_from_loader("genesis_toolbox", loader)
    if spec is None:
        raise ImportError(f"Could not build spec for {_TOOLBOX_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["genesis_toolbox"] = mod
    try:
        loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("genesis_toolbox", None)
        raise
    return mod
