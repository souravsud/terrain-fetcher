import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require a live internet connection.",
    )
    parser.addoption(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a terrain-fetcher YAML config file.  When provided together "
            "with --integration the test suite will also run the WorldCover / "
            "roughness-map validation for the coordinates defined in that config "
            "(lat/lon or csv)."
        ),
    )
