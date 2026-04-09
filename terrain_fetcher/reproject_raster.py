from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from pyproj import Transformer
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