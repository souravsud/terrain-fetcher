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
from rasterio.warp import reproject as _warp_reproject, Resampling
from shapely.geometry import Polygon

from .reproject_raster import get_utm_crs, reproject_raster_to_utm, save_utm_metadata
from .lc_table import load_custom_landcover_table

R = 6_371_000.0  # Earth's mean radius in meters

_WORLDCOVER_TILE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
)
_WORLDCOVER_GRID_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/{version}/{year}/esa_worldcover_{year}_grid.geojson"
)

# ETH Global Canopy Height 2020 — 3°×3° Cloud-Optimised GeoTIFFs
# Lang et al. (2023), 10 m resolution, values in metres (uint8; 255 = nodata).
_ETH_CANOPY_URL = (
    "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Fn"
    "/download?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_{tile}_Map.tif"
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


def _canopy_tile_names(bounds: list) -> list[str]:
    """Return ETH canopy-height tile names covering *bounds*.

    Tiles are 3°×3° aligned to multiples of 3°.  Names follow the same
    ``{N/S}{lat:02d}{E/W}{lon:03d}`` convention used by ESA WorldCover
    (SW corner of each tile).

    Parameters
    ----------
    bounds:
        ``[min_lon, min_lat, max_lon, max_lat]`` in WGS-84 decimal degrees.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    tiles: list[str] = []
    lat_start = int(math.floor(min_lat / 3)) * 3
    lon_start = int(math.floor(min_lon / 3)) * 3
    lat = lat_start
    while lat < max_lat:
        lon = lon_start
        while lon < max_lon:
            lat_hem = "N" if lat >= 0 else "S"
            lon_hem = "E" if lon >= 0 else "W"
            tile = f"{lat_hem}{abs(lat):02d}{lon_hem}{abs(lon):03d}"
            tiles.append(tile)
            lon += 3
        lat += 3
    return tiles


def stitch_canopy_tiles(bounds: list) -> tuple[np.ndarray | None, dict | None]:
    """Download and stitch ETH Global Canopy Height tiles, then crop to *bounds*.

    Returns ``(data_float32_m, profile)`` on success, or ``(None, None)`` when
    no tiles are available for the requested area (e.g. over open ocean).

    Parameters
    ----------
    bounds:
        ``[min_lon, min_lat, max_lon, max_lat]`` in WGS-84 decimal degrees.
    """
    tiles = _canopy_tile_names(bounds)
    paths: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for tile in tiles:
            url = _ETH_CANOPY_URL.format(tile=tile)
            try:
                r = requests.get(url, timeout=120)
            except requests.RequestException:
                continue
            if r.status_code == 200:
                p = tmp_path / f"{tile}.tif"
                p.write_bytes(r.content)
                paths.append(str(p))

        if not paths:
            return None, None

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
            prof.update(
                height=data.shape[0],
                width=data.shape[1],
                transform=tf,
                dtype=rasterio.float32,
                nodata=255,
                count=1,
            )

    # Replace ETH nodata (255) with NaN and convert to float32
    data_f = data.astype(np.float32)
    data_f[data_f == 255] = np.nan
    return data_f, prof


def _align_to_reference(
    src_data: np.ndarray,
    src_profile: dict,
    ref_profile: dict,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Resample *src_data* to exactly match the *ref_profile* grid.

    Uses bilinear interpolation by default so that continuous canopy-height
    values avoid the stair-step artefacts that nearest-neighbour would produce.

    Parameters
    ----------
    src_data:
        Source 2-D float32 array.
    src_profile:
        Rasterio profile dict for *src_data* (must contain ``crs`` and
        ``transform``).
    ref_profile:
        Rasterio profile dict that defines the target grid (``crs``,
        ``transform``, ``width``, ``height``).
    resampling:
        Rasterio ``Resampling`` enum value.  Defaults to bilinear.

    Returns
    -------
    np.ndarray
        Float32 array aligned to *ref_profile*.
    """
    dst = np.full(
        (ref_profile["height"], ref_profile["width"]),
        fill_value=np.nan,
        dtype=np.float32,
    )
    _warp_reproject(
        source=src_data.astype(np.float32),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=resampling,
    )
    return dst


def _compute_ora_z0_d(
    data_lc: np.ndarray,
    h: np.ndarray | None,
    lct: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute z0 and displacement-height (d) arrays using the direct ORA model.

    The ORA (Obstacle Roughness Assessment) model relates canopy height *h* to
    aerodynamic parameters:

    * Tree pixel with valid *h* (h > 0, finite):
      ``z0 = 0.1 × h``,  ``d = (2/3) × h``
    * Tree pixel with missing/zero *h*:
      ``z0 = table[10].z0`` (typically 1.5 m),  ``d = 0.0``
    * Non-tree pixel:
      ``z0 = lct[class_code].z0``,  ``d = 0.0``

    Hard physical caps are applied after calculation:
      ``z0 ≤ 3.0 m``,  ``d ≤ 25.0 m``

    Parameters
    ----------
    data_lc:
        2-D uint8 ESA WorldCover class-code array.
    h:
        2-D float32 canopy-height array aligned to *data_lc*, or ``None``
        when no ETH canopy data was available.
    lct:
        Land-cover table dict ``{int → {"z0": float, …}}``.

    Returns
    -------
    (z0_data, d_data)
        Both arrays are float32 with the same shape as *data_lc*.
    """
    # Build z0 lookup for non-tree pixels from the configured table
    lc_code_to_z0 = {
        lc_id: float(params.get("z0", 0.0))
        for lc_id, params in lct.items()
        if params is not None
    }
    # Fallback z0 for tree pixels when height is unavailable (GWA4 class 10 = 1.5)
    tree_fallback_z0 = lc_code_to_z0.get(10, 1.5)

    # Vectorised lookup for non-tree z0 baseline
    z0_lookup = np.vectorize(lambda c: lc_code_to_z0.get(int(c), np.nan))(
        data_lc
    ).astype(np.float32)

    is_tree = data_lc == 10

    if h is not None:
        h = np.asarray(h, dtype=np.float32)
        has_valid_h = is_tree & (h > 0) & np.isfinite(h)
        is_tree_no_h = is_tree & ~has_valid_h
    else:
        h = np.zeros_like(data_lc, dtype=np.float32)  # sentinel; never selected
        has_valid_h = np.zeros_like(is_tree, dtype=bool)
        is_tree_no_h = is_tree

    z0_data = np.where(
        has_valid_h, 0.10 * h,
        np.where(is_tree_no_h, tree_fallback_z0, z0_lookup),
    ).astype(np.float32)

    d_data = np.where(
        has_valid_h, (2.0 / 3.0) * h,
        0.0,
    ).astype(np.float32)

    # Physical caps
    z0_data = np.clip(z0_data, 0.0, 3.0)
    d_data = np.clip(d_data, 0.0, 25.0)

    return z0_data, d_data


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
    ) -> tuple[str, str | None, str | None]:
    """Download DEM and optional roughness map for a square area around a point.

    Parameters
    ----------
    index:
        Output file index (used for naming).
    center_lon, center_lat:
        Centre coordinates in decimal degrees (WGS-84).
    config:
        A :class:`~terrain_fetcher.download_config.DownloadConfig` instance.
    out_dir:
        Directory where output files are written.

    Returns
    -------
    (dem_file, roughness_file | None, displacement_file | None)
        *roughness_file* and *displacement_file* are ``None`` when
        ``config.include_roughness_map`` is ``False``.
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
    dmap_out_file = None
    if config.include_roughness_map:
        rmap_out_file = _generate_filename(
            index, center_lat, center_lon, out_dir, side_km, "worldcover", "roughness"
        )
        dmap_out_file = _generate_filename(
            index, center_lat, center_lon, out_dir, side_km, "worldcover", "displacement"
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

        lc_table_name = config.land_cover_table
        if lc_table_name == "custom":
            if not config.custom_land_cover_table_path:
                raise ValueError(
                    "land_cover_table is set to 'custom' but custom_land_cover_table_path "
                    "is not set in the config."
                )
            lct = load_custom_landcover_table(config.custom_land_cover_table_path)
            source = (
                f"{grid_url},{version},{year},custom:{config.custom_land_cover_table_path}"
            )
        else:
            lct = wk.get_landcover_table(lc_table_name)
            source = f"{grid_url},{version},{year},{lc_table_name}"

        # ----- ETH canopy height -----
        if verbose:
            print("Downloading ETH canopy height data...")
        data_canopy, profile_canopy = stitch_canopy_tiles(bounds)

        if data_canopy is not None:
            if verbose:
                print("Aligning canopy height to WorldCover grid (bilinear)...")
            h = _align_to_reference(data_canopy, profile_canopy, profile_lc)
        else:
            if verbose:
                print("No ETH canopy tiles available – tree pixels will use table fallback.")
            h = None

        if verbose:
            print("Applying ORA model to compute z0 and displacement height...")

        z0_data, d_data = _compute_ora_z0_d(data_lc, h, lct)

        profile_lc.update(dtype=rasterio.float32, count=1)

        if config.save_raw_files:
            raw_rmap_file = rmap_out_file.parent / f"{rmap_out_file.stem}_raw.tif"
            with rasterio.open(raw_rmap_file, "w", **profile_lc) as dst:
                dst.write(z0_data, 1)
            print(f"Saved raw roughness map (EPSG:4326) to: {raw_rmap_file.resolve()}")

            raw_dmap_file = dmap_out_file.parent / f"{dmap_out_file.stem}_raw.tif"
            with rasterio.open(raw_dmap_file, "w", **profile_lc) as dst:
                dst.write(d_data, 1)
            print(f"Saved raw displacement map (EPSG:4326) to: {raw_dmap_file.resolve()}")

        z0_data_utm, profile_z0_utm = reproject_raster_to_utm(
            z0_data, profile_lc, utm_crs, verbose
        )

        with rasterio.open(rmap_out_file, "w", **profile_z0_utm) as dst:
            dst.write(z0_data_utm, 1)

        print(f"Saved roughness map (UTM) to: {rmap_out_file.resolve()}")
        save_utm_metadata(source, rmap_out_file, center_lat, center_lon, profile_z0_utm)

        d_data_utm, profile_d_utm = reproject_raster_to_utm(
            d_data, profile_lc, utm_crs, verbose
        )

        with rasterio.open(dmap_out_file, "w", **profile_d_utm) as dst:
            dst.write(d_data_utm, 1)

        print(f"Saved displacement map (UTM) to: {dmap_out_file.resolve()}")
        save_utm_metadata(source, dmap_out_file, center_lat, center_lon, profile_d_utm)

        if config.show_plots:
            _plot_map(z0_data_utm, profile_z0_utm, side_km, "Roughness", out_dir)
            _plot_map(d_data_utm, profile_d_utm, side_km, "Displacement", out_dir)

    return (
        str(dem_out_file.resolve()),
        str(rmap_out_file.resolve()) if rmap_out_file else None,
        str(dmap_out_file.resolve()) if dmap_out_file else None,
    )
