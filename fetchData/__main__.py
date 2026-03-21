try:
    from .cli import main
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from fetchData.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
