from dataclasses import dataclass


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

    # Output options
    save_raw_files: bool = True

    # Debug options
    verbose: bool = True
    show_plots: bool = False