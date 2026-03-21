"""Command-line entry point for standalone terrain data downloads."""

from __future__ import annotations

import argparse
from pathlib import Path

from .download_config import DownloadConfig
from .csv_utils import load_coordinates_from_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and preprocess DEM/roughness raster data.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--csv", help="Path to a CSV with required columns: lat, lon")
    source_group.add_argument("--lat", type=float, help="Latitude for a single download")

    parser.add_argument("--lon", "--log", dest="lon", type=float, help="Longitude for a single download")
    parser.add_argument(
        "--output-root",
        default="out",
        help="Root folder where location-specific output directories are created",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index used for naming outputs (applies to single and batch modes)",
    )

    parser.add_argument("--dem-name", default="glo_30", help="DEM source name")
    parser.add_argument(
        "--side-km",
        type=float,
        default=50.0,
        help="Side length in kilometers for square extraction around each point",
    )
    parser.add_argument(
        "--area-or-point",
        choices=["Area", "Point"],
        default="Point",
        help="AREA_OR_POINT tag written into output DEM metadata",
    )
    parser.add_argument(
        "--ellipsoidal-height",
        action="store_true",
        help="Use ellipsoidal heights while requesting DEM data",
    )
    parser.add_argument(
        "--roughness-map",
        action="store_true",
        help="Also fetch and generate the aerodynamic roughness map",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Save debug plot PNGs for generated rasters",
    )
    parser.add_argument(
        "--save-raw-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to also save EPSG:4326 raw rasters (default: true)",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose logging (default: true)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first failed coordinate in batch mode",
    )

    return parser


def _build_config(args: argparse.Namespace) -> DownloadConfig:
    return DownloadConfig(
        dem_name=args.dem_name,
        dst_ellipsoidal_height=args.ellipsoidal_height,
        dst_area_or_point=args.area_or_point,
        side_length_km=args.side_km,
        include_roughness_map=args.roughness_map,
        verbose=args.verbose,
        show_plots=args.show_plots,
        save_raw_files=args.save_raw_files,
    )


def _resolve_coordinates(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.csv:
        return load_coordinates_from_csv(args.csv, verbose=args.verbose)

    if args.lon is None:
        raise ValueError("--lon is required when --lat is provided")

    return [(args.lat, args.lon)]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        coordinates = _resolve_coordinates(args)
    except ValueError as exc:
        parser.error(str(exc))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _build_config(args)

    # Import raster download utilities only when execution is requested.
    from .download_raster import DEMDownloader, create_output_dir
    downloader = DEMDownloader(config)

    success_count = 0
    failure_count = 0

    for offset, (lat, lon) in enumerate(coordinates):
        index = args.start_index + offset
        out_dir = create_output_dir(lat, lon, index, str(output_root))

        if out_dir is None:
            print(f"Skipping index={index}: output directory already exists for lat={lat}, lon={lon}")
            continue

        try:
            dem_file, roughness_file = downloader.download_single_location(lat, lon, index, out_dir)
            success_count += 1
            print(f"Completed index={index}")
            print(f"DEM: {dem_file}")
            if roughness_file:
                print(f"Roughness: {roughness_file}")
        except Exception as exc:  # noqa: BLE001
            failure_count += 1
            print(f"Failed index={index} (lat={lat}, lon={lon}): {exc}")
            if args.fail_fast:
                return 1

    if failure_count > 0:
        print(f"Finished with failures. Success={success_count}, Failed={failure_count}")
        return 1

    print(f"Finished successfully. Processed={success_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
