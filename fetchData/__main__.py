"""Enables: python -m fetchData [config.yaml]

Delegates to the root ``main.py`` entry point.
"""

import os
import sys

# Ensure the repository root (parent of this package) is on the path so that
# the root ``main`` module can be imported regardless of how Python was invoked.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
