"""Test setup for landsat_toolbox.pyt.

The toolbox imports `arcpy` at module load time. To exercise the pure-numpy
parts (persistence, algorithms, selectors) without an ArcGIS Pro install,
we stub the arcpy surface used at import and execute the .pyt file via
importlib. Tests then access classes/functions through `_toolbox`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _install_arcpy_stub() -> None:
    arcpy = ModuleType("arcpy")
    arcpy.AddMessage = lambda msg: None
    arcpy.AddWarning = lambda msg: None
    arcpy.AddError = lambda msg: None
    arcpy.SetLogHistory = lambda flag=False: None
    arcpy.SetLogMetadata = lambda flag=False: None
    arcpy.CheckExtension = lambda name: "Available"
    arcpy.CheckOutExtension = lambda name: None
    arcpy.CheckInExtension = lambda name: None
    arcpy.Exists = lambda path: False
    arcpy.Raster = lambda path: None
    arcpy.Describe = lambda path: SimpleNamespace(bandCount=1)
    arcpy.Point = lambda x, y: SimpleNamespace(X=x, Y=y)
    arcpy.RasterToNumPyArray = lambda *a, **kw: None
    arcpy.NumPyArrayToRaster = lambda *a, **kw: None
    arcpy.DefineProjection_management = lambda *a, **kw: None
    arcpy.env = SimpleNamespace(
        scratchFolder="",
        scratchWorkspace=None,
        workspace=None,
        overwriteOutput=False,
    )

    class _Filter:
        """Stub for arcpy.Parameter.filter — supports the .list attribute
        and the .type attribute used by GP filter configuration."""
        def __init__(self):
            self.list = []
            self.type = None

    class _Param:
        """Stub for arcpy.Parameter — auto-provides .filter, .value,
        .valueAsText, .altered, and accepts setErrorMessage/setWarningMessage
        calls so getParameterInfo / updateParameters / updateMessages can
        run without a real ArcGIS Pro environment."""
        def __init__(self, *a, **kw):
            self.filter = _Filter()
            self.value = None
            self.valueAsText = None
            self.values = None
            self.altered = False
            self.enabled = True
            for k, v in kw.items():
                setattr(self, k, v)

        def setErrorMessage(self, msg):
            self._error = msg

        def setWarningMessage(self, msg):
            self._warning = msg

        def clearMessage(self):
            self._error = None
            self._warning = None

    arcpy.Parameter = _Param

    management = ModuleType("arcpy.management")
    management.CompositeBands = lambda *a, **kw: None
    management.Delete = lambda *a, **kw: None
    arcpy.management = management

    sa = ModuleType("arcpy.sa")
    for name in ("Float", "Divide", "Times", "Con", "SetNull", "Plus", "Minus",
                 "PrincipalComponents", "ExtractByMask"):
        setattr(sa, name, lambda *a, **kw: None)
    arcpy.sa = sa

    ia = ModuleType("arcpy.ia")
    ia.ExtractBand = lambda *a, **kw: None
    ia.TransposeBits = lambda *a, **kw: None
    ia.GeometricMedian = lambda *a, **kw: None
    arcpy.ia = ia

    sys.modules["arcpy"] = arcpy
    sys.modules["arcpy.management"] = management
    sys.modules["arcpy.sa"] = sa
    sys.modules["arcpy.ia"] = ia


_install_arcpy_stub()

_HERE = os.path.dirname(os.path.abspath(__file__))
_GENESIS_PYT = os.path.normpath(os.path.join(_HERE, os.pardir, "genesis_toolbox.pyt"))


def _load_pyt(module_name: str, path: str):
    """.pyt is not a recognised Python source extension; force SourceFileLoader."""
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader(module_name, path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def genesis():
    """Loaded genesis_toolbox module (the unified multisensor toolbox)
    with arcpy stubbed."""
    return _load_pyt("genesis_toolbox", _GENESIS_PYT)
