"""Root entry point for terrain-fetcher.

Usage
-----
    python main.py [config.yaml]

All run settings and download options are read from the YAML config file.
See ``config.yaml`` in this directory for a fully annotated example.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download terrain DEM and roughness data. "
            "All settings are read from a YAML config file."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    import yaml

    with config_path.open() as fh:
        raw: dict = yaml.safe_load(fh) or {}

    # --- Run settings (read directly from YAML) ---
    csv_path = raw.get("csv")
    lat = raw.get("lat")
    lon = raw.get("lon")
    output_root = Path(raw.get("output_root", "out"))
    start_index = int(raw.get("start_index", 0))

    # --- Download settings (loaded via DownloadConfig) ---
    from terrain_fetcher.config import load_config
    from terrain_fetcher.csv_utils import load_coordinates_from_csv

    cfg = load_config(config_path)

    # --- Resolve coordinate list (validated before loading heavy deps) ---
    if csv_path:
        coordinates = load_coordinates_from_csv(csv_path, verbose=cfg.verbose)
    elif lat is not None and lon is not None:
        coordinates = [(float(lat), float(lon))]
    else:
        print(
            "Error: config must specify either 'csv' or both 'lat' and 'lon'.",
            file=sys.stderr,
        )
        return 1

    # --- Import raster download utilities only when coordinates are confirmed ---
    from terrain_fetcher.download_raster import DEMDownloader, create_output_dir

    output_root.mkdir(parents=True, exist_ok=True)
    downloader = DEMDownloader(cfg)

    success_count = 0
    failure_count = 0

    for offset, (coord_lat, coord_lon) in enumerate(coordinates):
        index = start_index + offset
        out_dir = create_output_dir(coord_lat, coord_lon, index, str(output_root))

        if out_dir is None:
            print(
                f"Skipping index={index}: output already exists "
                f"for lat={coord_lat}, lon={coord_lon}"
            )
            continue

        try:
            dem_file, roughness_file = downloader.download_single_location(
                coord_lat, coord_lon, index, out_dir
            )
            success_count += 1
            print(f"Completed index={index}")
            print(f"  DEM: {dem_file}")
            if roughness_file:
                print(f"  Roughness: {roughness_file}")
        except Exception as exc:  # noqa: BLE001
            failure_count += 1
            print(
                f"Failed index={index} (lat={coord_lat}, lon={coord_lon}): "
                f"{type(exc).__name__}: {exc}"
            )

    if failure_count > 0:
        print(f"Finished with failures. Success={success_count}, Failed={failure_count}")
        return 1

    print(f"Finished successfully. Processed={success_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

