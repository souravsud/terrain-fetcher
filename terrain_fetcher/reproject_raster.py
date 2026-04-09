from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from pyproj import Transformer
import json
from pathlib import Path
import numpy as np

def get_utm_crs(longitude: float, latitude: float) -> CRS:
    """Determine appropriate UTM CRS for given coordinates"""
    utm_zone = int((longitude + 180) / 6) + 1
    if latitude >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
    return CRS.from_epsg(epsg_code)

def reproject_raster_to_utm(data: np.ndarray, profile: dict, utm_crs: CRS, verbose: bool = False):
    """
    Reproject raster data from EPSG:4326 to UTM
    Returns:
        (reprojected_data, reprojected_profile)
    """
    src_crs = profile['crs']
    
    if verbose:
        print(f"Reprojecting from {src_crs} to {utm_crs}")
    
    # Calculate transform and dimensions for UTM
    bounds = array_bounds(profile['height'], profile['width'], profile['transform'])
    transform, width, height = calculate_default_transform(
        src_crs, 
        utm_crs, 
        profile['width'], 
        profile['height'],
        *bounds
    )
    
    # Update profile for UTM
    utm_profile = profile.copy()
    utm_profile.update({
        'crs': utm_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    
    # Create output array
    reprojected_data = np.empty((height, width), dtype=data.dtype)
    
    # Perform reprojection
    reproject(
        source=data,
        destination=reprojected_data,
        src_transform=profile['transform'],
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=utm_crs,
        resampling=Resampling.bilinear
    )
    
    if verbose:
        print(f"Reprojected to UTM: {width}x{height} pixels")
    
    return reprojected_data, utm_profile

def _section_from_profile(profile_utm: dict) -> dict:
    """Extract bounds and resolution metadata from a UTM-reprojected rasterio profile."""
    bounds_utm = array_bounds(
        profile_utm['height'],
        profile_utm['width'],
        profile_utm['transform'],
    )
    return {
        "bounds_utm": list(bounds_utm),
        "resolution_m": profile_utm['transform'].a,
    }


def save_combined_metadata(
    output_file: Path,
    center_lat: float,
    center_lon: float,
    side_km: float,
    utm_crs,
    terrain: dict,
    roughness: dict | None = None,
    displacement: dict | None = None,
):
    """Save combined metadata for terrain (and optional roughness/displacement) to JSON.

    Args:
        output_file: Base path; ``.json`` replaces the suffix.
        center_lat: Centre latitude in decimal degrees (WGS-84).
        center_lon: Centre longitude in decimal degrees (WGS-84).
        side_km: Side length of the square domain in kilometres.
        utm_crs: Rasterio/pyproj CRS object for the output UTM projection.
        terrain: Per-product metadata dict for the terrain/DEM layer.
        roughness: Per-product metadata dict for the roughness (z0) layer, or None.
        displacement: Per-product metadata dict for the displacement-height layer, or None.
    """
    # Calculate centre coordinates in UTM
    transformer = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
    center_utm_x, center_utm_y = transformer.transform(center_lon, center_lat)

    metadata: dict = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "side_km": side_km,
        "center_utm": [center_utm_x, center_utm_y],
        "utm_zone": utm_crs.to_string(),
        "epsg": utm_crs.to_epsg(),
        "crs": "UTM",
        "terrain": terrain,
    }
    if roughness is not None:
        metadata["roughness"] = roughness
    if displacement is not None:
        metadata["displacement"] = displacement

    metadata_file = output_file.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_file}")