# terrain-fetcher

Installable extraction of the `terrain_fetcher` package

## What this package provides

- DEM download and stitching
- Roughness-map generation from ESA WorldCover
- Raster reprojection to local UTM CRS
- CSV coordinate loading helpers
- YAML-based configuration loading

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
python -m terrain_fetcher config.yaml
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
from terrain_fetcher import DownloadConfig, create_output_dir, download_raster_data, load_config
from terrain_fetcher.csv_utils import load_coordinates_from_csv
from terrain_fetcher.parameter_generation import generate_directions

cfg = load_config("config.yaml")
```