"""YAML configuration loader for terrain-fetcher.

Usage
-----
Create a YAML file (e.g. ``config.yaml``) and pass it to ``load_config``::

    from terrain_fetcher.config import load_config

    cfg = load_config("config.yaml")

The returned :class:`~terrain_fetcher.download_config.DownloadConfig` object can be
passed directly to :class:`~terrain_fetcher.download_raster.DEMDownloader` or to the
CLI via ``--config config.yaml``.

Supported YAML keys (all optional; defaults match ``DownloadConfig`` defaults)
------------------------------------------------------------------------------
dem_name                    : str   – DEM source name (default: "glo_30")
ellipsoidal_height          : bool  – use ellipsoidal heights (default: false)
area_or_point               : str   – "Area" or "Point" (default: "Point")
side_km                     : float – square side length in km (default: 50.0)
roughness_map               : bool  – generate roughness map (default: false)
worldcover_version          : str   – ESA WorldCover version tag (default: "v100")
worldcover_year             : int   – ESA WorldCover data year (default: 2020)
land_cover_table            : str   – windkit land-cover table name, or "custom" to
                                      load from a user-supplied CSV file (default: "GWA4")
custom_land_cover_table_path: str   – path to the custom land-cover CSV file; only
                                      used when land_cover_table is "custom".  Defaults
                                      to ``landcover_roughness.csv`` in the same directory
                                      as this config file.
save_raw_files              : bool  – save EPSG:4326 raw rasters (default: true)
verbose                     : bool  – verbose logging (default: true)
show_plots                  : bool  – save debug plot PNGs (default: false)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .download_config import DownloadConfig

_DEFAULT_CUSTOM_TABLE_FILENAME = "landcover_roughness.csv"


def load_config(yaml_path: str | Path) -> DownloadConfig:
    """Load a :class:`DownloadConfig` from a YAML file.

    Parameters
    ----------
    yaml_path:
        Path to the YAML configuration file.

    Returns
    -------
    DownloadConfig
        A fully populated config object.  Any key absent from the YAML file
        falls back to the ``DownloadConfig`` field default.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    ValueError
        If the YAML file cannot be parsed or contains an invalid value.
    """
    import yaml  # soft dependency; already required via pyproject.toml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level, got {type(data).__name__}")

    land_cover_table: str = data.get("land_cover_table", "GWA4")

    # Resolve the custom table path when land_cover_table == "custom".
    custom_path: str | None = None
    if land_cover_table == "custom":
        raw_custom_path = data.get("custom_land_cover_table_path")
        if raw_custom_path is not None:
            custom_path = str(Path(raw_custom_path))
        else:
            # Default: sibling file next to the config YAML
            custom_path = str(yaml_path.parent / _DEFAULT_CUSTOM_TABLE_FILENAME)

    return DownloadConfig(
        dem_name=data.get("dem_name", "glo_30"),
        dst_ellipsoidal_height=data.get("ellipsoidal_height", False),
        dst_area_or_point=data.get("area_or_point", "Point"),
        side_length_km=float(data.get("side_km", 50.0)),
        include_roughness_map=data.get("roughness_map", False),
        worldcover_version=data.get("worldcover_version", "v100"),
        worldcover_year=int(data.get("worldcover_year", 2020)),
        land_cover_table=land_cover_table,
        custom_land_cover_table_path=custom_path,
        save_raw_files=data.get("save_raw_files", True),
        verbose=data.get("verbose", True),
        show_plots=data.get("show_plots", False),
    )
