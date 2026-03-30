from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DownloadConfig:
    # DEM parameters
    dem_name: str = "glo_30"
    dst_ellipsoidal_height: bool = False
    dst_area_or_point: str = "Point"
    side_length_km: float = 50.0

    # Roughness map parameters
    include_roughness_map: bool = False
    worldcover_version: str = "v100"
    worldcover_year: int = 2020
    land_cover_table: str = "GWA4"
    # Path to a custom land-cover → z0 CSV file.
    # Required when land_cover_table == "custom"; ignored otherwise.
    custom_land_cover_table_path: str | None = field(default=None)

    # Output options
    save_raw_files: bool = True

    # Debug options
    verbose: bool = True
    show_plots: bool = False