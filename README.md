# cfd-fetchdata

Installable extraction of the `fetchData` package from the CFD dataset pipeline.

## What this package provides

- DEM download and stitching
- Roughness-map generation from ESA WorldCover
- Raster reprojection to local UTM CRS
- CSV coordinate loading helpers

## Package layout

```text
fetchData/
  __init__.py
  csv_utils.py
  download_config.py
  download_raster.py
  parameter_generation.py
  reproject_raster.py
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

## Consumer usage

```python
from fetchData import DownloadConfig, create_output_dir, download_raster_data
from fetchData.csv_utils import load_coordinates_from_csv
from fetchData.parameter_generation import generate_directions
```

## Standalone CLI usage

Run with Python directly from the repository root.

Single location:

```bash
python fetchData --lat 39.71121111 --lon -7.73483333 --output-root out --side-km 50 --roughness-map
```

Batch from CSV (must contain `lat` and `lon` columns):

```bash
python fetchData --csv coordinates.csv --output-root out --start-index 0 --no-verbose
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
