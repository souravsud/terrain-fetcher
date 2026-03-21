# cfd-fetchdata

Installable extraction of the `fetchData` package from the CFD dataset pipeline.

## What this package provides

- DEM download and stitching
- Roughness-map generation from ESA WorldCover
- Raster reprojection to local UTM CRS
- CSV coordinate loading helpers
- Wind direction generation helpers
- YAML-based configuration loading

## Package layout

```text
fetchData/
  __init__.py
  config.py           ← YAML config loader
  csv_utils.py
  download_config.py
  download_raster.py
  parameter_generation.py
  reproject_raster.py
config.yaml           ← example / default configuration file
main.py               ← root entry point
```

## Installation

From this folder:

```bash
pip install .
```

For active development:

```bash
pip install -e .
```

## Running

All settings — coordinate source, output path, DEM options, etc. — live in a
single YAML file.  Edit `config.yaml` (or copy it) and run:

```bash
python main.py config.yaml
```

If you omit the argument, `config.yaml` in the current directory is used:

```bash
python main.py
```

You can also invoke the package directly (equivalent):

```bash
python -m fetchData config.yaml
```

## Configuration

Copy and edit the bundled `config.yaml`:

```yaml
# Coordinate source — one of: csv  OR  lat + lon
# csv: coordinates.csv         # path to CSV with 'lat' and 'lon' columns
lat: 39.71121111               # single-location latitude
lon: -7.73483333               # single-location longitude

# Run settings
output_root: out               # root folder for per-location output directories
start_index: 0                 # starting index for output file/folder naming

# DEM settings
dem_name: glo_30               # DEM dataset (e.g. glo_30, nasadem)
ellipsoidal_height: false
area_or_point: Point           # "Area" or "Point"
side_km: 50.0                  # extraction window side length in km

# Roughness map
roughness_map: false           # also generate aerodynamic roughness map
worldcover_version: v100
worldcover_year: 2020
land_cover_table: GWA4

# Output / debug
save_raw_files: true
verbose: true
show_plots: false
```

All keys are optional; omitted keys fall back to the `DownloadConfig` defaults.

## Programmatic usage

```python
from fetchData import DownloadConfig, create_output_dir, download_raster_data, load_config
from fetchData.csv_utils import load_coordinates_from_csv
from fetchData.parameter_generation import generate_directions

cfg = load_config("config.yaml")
```

## Notes

- The import package name remains `fetchData` to keep existing consumer code unchanged.
- The package now lives at the repository root instead of under `src/`.
- All CLI parsing lives in `main.py`; `fetchData/` is a pure library package.
