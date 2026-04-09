# Configuration Reference

All settings are read from a single YAML file (default: `config.yaml`).
Pass the path as the first argument to the CLI:

```bash
python main.py config.yaml
# or
python -m terrain_fetcher config.yaml
```

Every key is optional; omitted keys fall back to the `DownloadConfig` defaults shown below.

---

## Coordinate source

Provide **either** a CSV file **or** an explicit `lat`/`lon` pair.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `csv` | string | — | Path to a CSV file with `lat` and `lon` columns. When set, `lat`/`lon` keys are ignored. |
| `lat` | float | — | Single-location latitude (WGS-84 decimal degrees). |
| `lon` | float | — | Single-location longitude (WGS-84 decimal degrees). |

---

## Run settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_root` | string | `"out"` | Root folder; per-location subdirectories are created inside it. |
| `start_index` | int | `0` | Starting counter used in output folder and file names. |

---

## DEM settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dem_name` | string | `"glo_30"` | DEM dataset name understood by [dem-stitcher](https://github.com/opera-adt/dem-stitcher) (e.g. `glo_30`, `nasadem`). |
| `ellipsoidal_height` | bool | `false` | When `true`, request ellipsoidal heights instead of orthometric. |
| `area_or_point` | string | `"Point"` | Metadata tag written into output GeoTIFFs (`"Area"` or `"Point"`). |
| `side_km` | float | `50.0` | Side length of the square extraction window in kilometres. |

---

## Roughness map settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `roughness_map` | bool | `false` | When `true`, also download ESA WorldCover and convert it to a z0 roughness raster. Canopy height tiles are fetched automatically to apply the ORA model for tree pixels, and a co-registered displacement-height raster is produced. |
| `worldcover_version` | string | `"v100"` | ESA WorldCover release tag (e.g. `v100`, `v200`). |
| `worldcover_year` | int | `2020` | ESA WorldCover data year. |
| `land_cover_table` | string | `"GWA4"` | windkit land-cover lookup table used for z0 conversion (e.g. `"GWA4"`, `"GWA3"`; see [windkit documentation](https://windkit.readthedocs.io) for all valid names). Use `"custom"` to load from a user-supplied CSV file instead — see [custom-lookup-table.md](custom-lookup-table.md). |
| `custom_land_cover_table_path` | string | `"landcover_roughness.csv"` | Path to the custom land-cover CSV file. Only used when `land_cover_table: custom`. Defaults to `landcover_roughness.csv` in the same directory as the config file. |

---

## Output options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_raw_files` | bool | `true` | When `true`, also save EPSG:4326 (raw) rasters alongside the UTM output. |

---

## Debug / logging settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `verbose` | bool | `true` | Print progress messages to stdout. |
| `show_plots` | bool | `false` | Save PNG plots of generated rasters. |

---

## Full example

```yaml
# terrain-fetcher configuration file

# ---------------------------------------------------------------------------
# Coordinate source  (one of: csv  OR  lat + lon)
# ---------------------------------------------------------------------------
# csv: coordinates.csv        # path to a CSV file with 'lat' and 'lon' columns
lat: 39.71121111              # single-location latitude
lon: -7.73483333              # single-location longitude

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
output_root: out              # root folder for per-location output directories
start_index: 0                # starting index for output file/folder naming

# ---------------------------------------------------------------------------
# DEM settings
# ---------------------------------------------------------------------------
dem_name: glo_30              # DEM dataset name understood by dem-stitcher
ellipsoidal_height: false     # true → request ellipsoidal heights
area_or_point: Point          # metadata tag for output GeoTIFFs ("Area" or "Point")
side_km: 50.0                 # side length of the square extraction window in km

# ---------------------------------------------------------------------------
# Roughness map settings
# ---------------------------------------------------------------------------
roughness_map: false          # true → download WorldCover + canopy height,
                              #         generate z0 and displacement-height rasters
worldcover_version: v100      # ESA WorldCover release tag (e.g. v100, v200)
worldcover_year: 2020         # ESA WorldCover data year
land_cover_table: GWA4        # windkit lookup table; use "custom" for a user CSV
# custom_land_cover_table_path: landcover_roughness.csv  # only when land_cover_table: custom

# ---------------------------------------------------------------------------
# Output options
# ---------------------------------------------------------------------------
save_raw_files: true          # true → also save EPSG:4326 raw rasters

# ---------------------------------------------------------------------------
# Debug / logging
# ---------------------------------------------------------------------------
verbose: true
show_plots: false
```
