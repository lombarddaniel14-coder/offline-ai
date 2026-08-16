"""Dev-only stand-in for src/portable.py (interface contract #1).

Used ONLY by tests while the GUI agent's real portable.py hasn't landed yet:
the test loader imports this file under the module name 'portable' when
`import portable` fails. Never shipped, never imported by production code.
"""

import sys
from pathlib import Path


def app_root() -> Path:
    """The 'Offline AI' folder root.

    Frozen: exe lives at <root>/app/TBSOffline/TBSOffline.exe
            -> root = exe.parents[2] (per contract).
    Source: this stub lives at <root>/_dev/portable_stub.py -> parents[1].
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parents[2]
    return Path(__file__).resolve().parents[1]
