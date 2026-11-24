"""Test package bootstrap: make the plugin importable as ``agy_acp`` from any checkout name.

``unittest discover`` imports this package first, so the path fix lands before any test module does
its own ``from agy_acp...`` import. Without it the suite only worked when the checkout directory
happened to be named ``agy_acp`` on ``sys.path``.
"""

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_PARENT = str(_PACKAGE_DIR.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
