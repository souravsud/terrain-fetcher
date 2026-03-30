"""Utilities for loading land-cover → roughness (z0) lookup tables."""

from __future__ import annotations

from pathlib import Path


def load_custom_landcover_table(path: str | Path) -> dict:
    """Load a user-supplied land-cover → z0 lookup table from a text/CSV file.

    The file format is whitespace- or comma-separated values with optional
    ``#``-prefixed comment lines and blank lines.  Each data row must contain
    at least two columns::

        class_id   z0   [description ...]

    Both comma-separated and space/tab-separated layouts are supported, so
    the following rows are equivalent:

    .. code-block:: text

        10,0.6,Tree cover
        10  0.6  Tree cover

    Parameters
    ----------
    path:
        Path to the land-cover table file.

    Returns
    -------
    dict
        Mapping ``{class_id (int): {"z0": float, "description": str}}``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file contains no valid data rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Custom land-cover table not found: {path}")

    table: dict = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Accept both comma-separated and whitespace-separated layouts
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                class_id = int(parts[0])
                z0 = float(parts[1])
            except ValueError:
                continue
            description = " ".join(parts[2:]) if len(parts) > 2 else ""
            table[class_id] = {"z0": z0, "description": description}

    if not table:
        raise ValueError(f"No valid data rows found in custom land-cover table: {path}")

    return table
