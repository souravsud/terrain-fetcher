from pathlib import Path
import math
import os
import json

R = 6_371_000.0  # Earth's mean radius in meters

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

def generate_filename(index,center_lat, center_lon,out_dir,side_km, source, prefix, include_source= True):
    
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    lat_str = format_coord(center_lat, is_lat=True, precision=3)
    lon_str = format_coord(center_lon, is_lat=False, precision=3)
    side_str = f"{side_km:.0f}km" if side_km % 1 == 0 else f"{side_km:.1f}km"
    
    if include_source:
        file_name = f"{prefix}_{(index+1):04d}_{source}_{lat_str}_{lon_str}_{side_str}.tif"
    else:
        file_name = f"{prefix}_{(index+1):04d}_{lat_str}_{lon_str}_{side_str}.tif"
    
    out_file = out_path / file_name
    
    return out_file

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

    metadata: dict = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "side_km": side_km,
        "utm_zone": utm_crs.to_string(),
        "epsg": utm_crs.to_epsg(),
        "crs": "UTM",
        "terrain": terrain,
    }
    if roughness is not None:
        metadata["roughness"] = roughness
    if displacement is not None:
        metadata["displacement"] = displacement

    metadata_file = output_file.parent / "terrain_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_file}")