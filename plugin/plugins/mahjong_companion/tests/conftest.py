from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `plugin.plugins.mahjong_companion...`
# imports work even when pytest is invoked from inside the plugin directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
