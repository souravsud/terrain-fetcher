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

## Configuration

All runtime settings live in a single `DownloadConfig` dataclass.  You can
populate it directly in Python, or load it from a YAML file.

### YAML configuration file

Copy and edit the bundled `config.yaml`:

```yaml
dem_name: glo_30            # DEM dataset (e.g. glo_30, nasadem)
ellipsoidal_height: false
area_or_point: Point        # "Area" or "Point"
side_km: 50.0               # extraction window side length in km

roughness_map: false        # also generate aerodynamic roughness map
worldcover_version: v100    # ESA WorldCover release tag
worldcover_year: 2020
land_cover_table: GWA4      # windkit land-cover lookup table

save_raw_files: true        # save EPSG:4326 raw rasters as well

verbose: true
show_plots: false
```

All keys are optional; omitted keys fall back to the `DownloadConfig` defaults.

### Loading in Python

```python
from fetchData.config import load_config

cfg = load_config("config.yaml")
```

## Consumer usage

```python
from fetchData import DownloadConfig, create_output_dir, download_raster_data, load_config
from fetchData.csv_utils import load_coordinates_from_csv
from fetchData.parameter_generation import generate_directions
```

## Standalone CLI usage

Run with Python directly from the repository root.

Single location:

```bash
python fetchData --lat 39.71121111 --lon -7.73483333 --output-root out --side-km 50 --roughness-map
```

Single location using a YAML config file:

```bash
python fetchData --lat 39.71121111 --lon -7.73483333 --config config.yaml
```

Batch from CSV (must contain `lat` and `lon` columns):

```bash
python fetchData --csv coordinates.csv --output-root out --start-index 0 --no-verbose
```

Batch from CSV with a YAML config (CLI flags override YAML values):

```bash
python fetchData --csv coordinates.csv --config config.yaml --no-verbose
```

Alternative module form:

```bash
python -m fetchData --lat 39.71121111 --lon -7.73483333
```

Alternative root script form:

```bash
python main.py --lat 39.71121111 --lon -7.73483333
```

## Notes

- The import package name remains `fetchData` to keep existing consumer code unchanged.
- The package now lives at the repository root instead of under `src/`.
- Source files are kept intentionally close to the original implementation for low-risk migration.
- CLI flags always take precedence over YAML config values.
