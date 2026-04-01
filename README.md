# terrain-fetcher

Installable extraction of the `terrain_fetcher` package

## What this package provides

- DEM download and stitching (via [dem-stitcher](https://github.com/opera-adt/dem-stitcher))
- Roughness-map generation from [ESA WorldCover](https://esa-worldcover.org)
- Canopy-height integration from [ETH Global Canopy Height 2020](https://langnico.github.io/globalcanopyheight/)
  with the ORA model applied to tree pixels
- Co-registered displacement-height raster output
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

Copy and edit the bundled `config.yaml`.  A minimal example:

```yaml
lat: 39.71121111
lon: -7.73483333
output_root: out
dem_name: glo_30
side_km: 50.0
roughness_map: true
land_cover_table: GWA4
```

All keys are optional; omitted keys fall back to the `DownloadConfig` defaults.

See **[docs/configuration.md](docs/configuration.md)** for the full list of
configuration keys and their defaults.

To use a custom land-cover → z0 lookup table, set `land_cover_table: custom` —
see **[docs/custom-lookup-table.md](docs/custom-lookup-table.md)**.

## Outputs

When `roughness_map: true` the pipeline produces the following rasters (in
addition to the DEM) inside each location folder:

| File | Description |
|------|-------------|
| `dem_utm.tif` | Digital elevation model reprojected to local UTM |
| `roughness_utm.tif` | Aerodynamic roughness length z0 \[m\] |
| `displacement_utm.tif` | Displacement height d \[m\] (tree pixels via ORA model; 0 for all other land-cover classes) |

When `save_raw_files: true`, the corresponding EPSG:4326 versions are saved
alongside the UTM rasters.

### Sample maps

<!-- TODO: upload a plot of all downloaded maps and the final roughness map -->

## Programmatic usage

```python
from terrain_fetcher import DownloadConfig, create_output_dir, download_raster_data, load_config
from terrain_fetcher.csv_utils import load_coordinates_from_csv

cfg = load_config("config.yaml")
```