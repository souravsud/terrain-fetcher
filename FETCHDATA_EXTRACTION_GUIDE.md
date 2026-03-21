# fetchData Extraction Guide

This repository now contains a standalone, installable extraction at:

- `extracted-fetchdata-repo/`

Use this guide to move it into a new git repository and clean up this project.

## What to move to the new repository

Move the entire folder:

- `extracted-fetchdata-repo/`

After moving, rename the destination repository root as desired (example: `cfd-fetchdata`).

## What to delete from this repository after successful migration

Delete the legacy in-repo package folder:

- `fetchData/`

Only delete this after the new package is installed in your environment and imports work.

## Install the extracted package

From the new repository root:

```bash
pip install -e .
```

## Verify imports in the main pipeline repo

The following existing imports should continue to work unchanged because the package name is still `fetchData`:

```python
from fetchData import download_raster_data, create_output_dir, DownloadConfig
from fetchData.csv_utils import load_coordinates_from_csv
from fetchData.parameter_generation import generate_directions
```

## Files included in the extracted package

- `pyproject.toml`
- `README.md`
- `fetchData/__init__.py`
- `fetchData/download_raster.py`
- `fetchData/download_config.py`
- `fetchData/reproject_raster.py`
- `fetchData/csv_utils.py`
- `fetchData/parameter_generation.py`
- `tests/test_imports.py`

## Recommended migration sequence

1. Create a new empty git repository.
2. Move the contents of `extracted-fetchdata-repo/` into that repository.
3. Commit and push the new repository.
4. Install it into your environment with `pip install -e .` (or pinned git URL).
5. In this repository, delete `fetchData/`.
6. Run your pipeline entrypoints to confirm imports and behavior.

## Optional post-migration change

After you push the new repository, replace this line in `environment.yml`:

```yaml
- -e ./extracted-fetchdata-repo
```

with a pinned git install line such as:

```yaml
- git+https://github.com/<your-user>/<new-repo>.git@v0.1.0
```
