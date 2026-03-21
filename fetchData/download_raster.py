import math
import tempfile
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
import windkit as wk
from dem_stitcher.stitcher import stitch_dem
from rasterio import plot
from rasterio.merge import merge
from shapely.geometry import Polygon

from .reproject_raster import get_utm_crs, reproject_raster_to_utm, save_utm_metadata

R = 6_371_000.0  # Earth's mean radius in meters

_WORLDCOVER_TILE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
)
_WORLDCOVER_GRID_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/{version}/{year}/esa_worldcover_{year}_grid.geojson"
)


class DEMDownloader:
    """Handles DEM downloading with configurable options."""

    def __init__(self, config):
        self.config = config

    def log(self, message):
        """Print *message* only when verbose mode is enabled."""
        if self.config.verbose:
            print(message)

    def download_single_location(self, lat, lon, index, out_dir):
        """Download DEM (and optionally roughness map) for a single coordinate pair."""
        self.log(f"Downloading DEM for lat={lat}, lon={lon}")
        return download_square_data(index=index, center_lon=lon, center_lat=lat, config=self.config, out_dir=out_dir)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def format_coord(value: float, is_lat: bool, precision: int = 5) -> str:
    """Format a latitude or longitude into a fixed-width, signed, filesystem-safe string."""
    if is_lat:
        hemi = "N" if value >= 0 else "S"
        width = 2
    else:
        hemi = "E" if value >= 0 else "W"
        width = 3

    abs_val = abs(value)
    deg = int(math.floor(abs_val))
    frac = abs_val - deg
    frac_int = int(round(frac * (10 ** precision)))

    deg_str = f"{deg:0{width}d}"
    frac_str = f"{frac_int:0{precision}d}"

    return f"{hemi}{deg_str}_{frac_str}"

def create_output_dir(lat: float, lon: float, index: int, root_folder: str) -> str:
    #Save folder for each location
    lat_str = format_coord(lat, is_lat=True, precision=3)
    lon_str = format_coord(lon, is_lat=False, precision=3)
    folder_name = f"terrain_{(index+1):04d}_{lat_str}_{lon_str}"
    download_path = os.path.join(root_folder, folder_name)
    
    if os.path.exists(download_path):
        return None
    else:
        os.makedirs(download_path)
        return download_path

def latlon_offset(lat: float, lon: float, dy_m: float, dx_m: float) -> tuple[float, float]:
    """Move a point northwards by dy_m meters and eastwards by dx_m meters."""
    dlat_rad = dy_m / R
    dlon_rad = dx_m / (R * math.cos(math.radians(lat)))

    new_lat = lat + math.degrees(dlat_rad)
    new_lon = lon + math.degrees(dlon_rad)
    return new_lat, new_lon

def stitch_tiles(tiles, version, year, bounds):
    """Download and stitch ESA WorldCover tiles, then crop to *bounds*."""
    paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for tile in tiles:
            url = _WORLDCOVER_TILE_URL.format(version=version, year=year, tile=tile)
            r = requests.get(url)
            if r.status_code == 200:
                p = tmp_path / f"{tile}.tif"
                p.write_bytes(r.content)
                paths.append(str(p))

        if not paths:
            raise ValueError("No WorldCover tiles downloaded")

        datasets = [rasterio.open(p) for p in paths]
        try:
            mosaic, transform = merge(datasets)
            prof = datasets[0].profile.copy()
        finally:
            for ds in datasets:
                ds.close()

        prof.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)

        mosaic_file = tmp_path / "mosaic.tif"
        with rasterio.open(mosaic_file, "w", **prof) as dst:
            dst.write(mosaic)

        with rasterio.open(mosaic_file) as src:
            win = rasterio.windows.from_bounds(*bounds, src.transform)
            data = src.read(1, window=win)
            tf = src.window_transform(win)
            prof.update(height=data.shape[0], width=data.shape[1], transform=tf)

    return data, prof

def _calculate_bounds(side_km, center_lat,center_lon):
    
    half = (side_km * 1000) / 2
    
    corners = [
        latlon_offset(center_lat, center_lon, +half, +half),
        latlon_offset(center_lat, center_lon, +half, -half),
        latlon_offset(center_lat, center_lon, -half, -half),
        latlon_offset(center_lat, center_lon, -half, +half),
    ]
    
    lats = [pt[0] for pt in corners]
    lons = [pt[1] for pt in corners]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    bounds = [min_lon, min_lat, max_lon, max_lat]

    return bounds, corners

def _generate_filename(index,center_lat, center_lon,out_dir,side_km, source, prefix):
    
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    lat_str = format_coord(center_lat, is_lat=True, precision=3)
    lon_str = format_coord(center_lon, is_lat=False, precision=3)
    side_str = f"{side_km:.0f}km" if side_km % 1 == 0 else f"{side_km:.1f}km"
    file_name = f"{prefix}_{(index+1):04d}_{source}_{lat_str}_{lon_str}_{side_str}.tif"
    
    out_file = out_path / file_name
    
    return out_file
        
def _plot_map(data, profile, side_km, plot_name, out_dir):
    cmap = "viridis" if plot_name == "Terrain" else "tab20"
    fig, ax = plt.subplots(figsize=(6, 6))
    plot.show(data, transform=profile["transform"], ax=ax, cmap=cmap)
    ax.set_title(f"{plot_name} {side_km}km square")
    ax.set_xlabel("Eastings (m)")
    ax.set_ylabel("Northings (m)")
    filename = f"{plot_name.lower().replace(' ', '_')}_map.png"
    plt.savefig(Path(out_dir) / filename)
    
def download_square_data(
        index: int,
        center_lon: float,
        center_lat: float,
        config,
        out_dir: str = "out",
    ) -> tuple[str, str | None]:
    """Download DEM and optional roughness map for a square area around a point.

    Parameters
    ----------
    index:
        Output file index (used for naming).
    center_lon, center_lat:
        Centre coordinates in decimal degrees (WGS-84).
    config:
        A :class:`~fetchData.download_config.DownloadConfig` instance.
    out_dir:
        Directory where output files are written.

    Returns
    -------
    (dem_file, roughness_file | None)
    """
    verbose = config.verbose
    side_km = config.side_length_km
    dem_name = config.dem_name

    bounds, corners = _calculate_bounds(side_km, center_lat, center_lon)

    if verbose:
        print(f"Bounds (lat/lon): {bounds}")

    # Determine UTM CRS once for both DEM and roughness
    utm_crs = get_utm_crs(center_lon, center_lat)

    if verbose:
        print(f"Target UTM CRS: {utm_crs}")

    dem_out_file = _generate_filename(index, center_lat, center_lon, out_dir, side_km, dem_name, "terrain")

    # ===== DEM PROCESSING =====
    if verbose:
        print("Downloading DEM data...")

    data, profile = stitch_dem(
        bounds,
        dem_name=dem_name,
        dst_ellipsoidal_height=config.dst_ellipsoidal_height,
        dst_area_or_point=config.dst_area_or_point,
    )

    if verbose:
        print(f"Original DEM CRS: {profile['crs']}")

    if config.save_raw_files:
        raw_dem_file = dem_out_file.parent / f"{dem_out_file.stem}_raw.tif"
        with rasterio.open(raw_dem_file, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(AREA_OR_POINT=config.dst_area_or_point)
        print(f"Saved raw DEM (EPSG:4326) to: {raw_dem_file.resolve()}")

    # Reproject to UTM
    data_utm, profile_utm = reproject_raster_to_utm(data, profile, utm_crs, verbose)

    with rasterio.open(dem_out_file, "w", **profile_utm) as dst:
        dst.write(data_utm, 1)
        dst.update_tags(AREA_OR_POINT=config.dst_area_or_point)

    print(f"Saved terrain elevation map (UTM) to: {dem_out_file.resolve()}")

    save_utm_metadata(dem_name, dem_out_file, center_lat, center_lon, profile_utm)

    if config.show_plots:
        _plot_map(data_utm, profile_utm, side_km, "Terrain", out_dir)

    # ===== ROUGHNESS MAP PROCESSING =====
    rmap_out_file = None
    if config.include_roughness_map:
        rmap_out_file = _generate_filename(
            index, center_lat, center_lon, out_dir, side_km, "worldcover", "roughness"
        )

        if verbose:
            print("Downloading roughness map data...")

        version = config.worldcover_version
        year = config.worldcover_year
        grid_url = _WORLDCOVER_GRID_URL.format(version=version, year=year)

        grid = gpd.read_file(grid_url)
        aoi = Polygon([(lon, lat) for lat, lon in corners])
        tiles = grid[grid.intersects(aoi)].ll_tile.tolist()

        if verbose:
            print(f"Tiles to download: {tiles}")

        data_lc, profile_lc = stitch_tiles(tiles, version, year, bounds)

        lct = wk.get_landcover_table(config.land_cover_table)
        source = f"{grid_url},{version},{year},{config.land_cover_table}"

        if verbose:
            print("Converting WorldCover classes to aerodynamic roughness length (z0)...")

        lc_code_to_z0 = {
            lc_id: params.get("z0")
            for lc_id, params in lct.items()
            if params is not None and "z0" in params
        }
        z0_data = np.vectorize(lc_code_to_z0.get)(data_lc)

        profile_lc.update(dtype=rasterio.float32, count=1)

        if config.save_raw_files:
            raw_rmap_file = rmap_out_file.parent / f"{rmap_out_file.stem}_raw.tif"
            with rasterio.open(raw_rmap_file, "w", **profile_lc) as dst:
                dst.write(z0_data.astype(np.float32), 1)
            print(f"Saved raw roughness map (EPSG:4326) to: {raw_rmap_file.resolve()}")

        z0_data_utm, profile_z0_utm = reproject_raster_to_utm(
            z0_data.astype(np.float32), profile_lc, utm_crs, verbose
        )

        with rasterio.open(rmap_out_file, "w", **profile_z0_utm) as dst:
            dst.write(z0_data_utm, 1)

        print(f"Saved roughness map (UTM) to: {rmap_out_file.resolve()}")

        save_utm_metadata(source, rmap_out_file, center_lat, center_lon, profile_z0_utm)

        if config.show_plots:
            _plot_map(z0_data_utm, profile_z0_utm, side_km, "Roughness", out_dir)

    return str(dem_out_file.resolve()), str(rmap_out_file.resolve()) if rmap_out_file else None
