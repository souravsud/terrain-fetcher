#!/usr/bin/env python3
"""Roughness map analysis script for terrain-fetcher outputs.

For each roughness map provided this script:

* Prints summary statistics (min, max, mean, median, standard deviation,
  fraction of valid pixels whose z0 exceeds the threshold).
* Saves an annotated PNG that shows the full roughness map alongside a
  binary mask of all pixels above the threshold.
* Reads the companion ``terrain_metadata.json`` (if present) to report the
  model parameters that were used when the map was produced.
* Performs a **source attribution** analysis that answers "was the high z0
  driven by canopy-height data or by the fixed land-cover table fallback?":

  - *Canopy-derived*  – tree pixels whose z0 was computed as
    ``0.1 × canopy_height``; these appear as non-round values.
  - *Fixed fallback*  – tree pixels for which no canopy height was available
    so the table fallback value (e.g. 1.5 m for GWA4) was used.

  When a ``_raw.tif`` roughness file exists next to the UTM file the
  attribution uses it (discrete table values are preserved exactly before
  UTM reprojection).  When only the UTM file is available the attribution
  falls back to a tolerance-based comparison against the fallback value
  extracted from the metadata.

Usage
-----
::

    # Analyse one file
    python analyze_roughness.py path/to/roughness_0001_worldcover_*.tif

    # Analyse several files
    python analyze_roughness.py roughness_a.tif roughness_b.tif

    # Scan an output directory for all roughness maps automatically
    python analyze_roughness.py --dir out/

    # Use a custom threshold (default 1.0 m)
    python analyze_roughness.py --threshold 0.5 roughness.tif

    # Write annotated PNGs to a specific folder
    python analyze_roughness.py --output-dir reports/ roughness.tif

Dependencies
------------
``rasterio``, ``numpy``, ``matplotlib``  – all already required by
terrain-fetcher's ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.plot import show as rplot

# ── ORA model constants (mirror of download_raster.py) ─────────────────────
_ORA_Z0_TREE_FACTOR = 0.10   # z0 = 0.1 × canopy_height for tree pixels
_ORA_Z0_MAX_CAP_M   = 3.0   # hard cap from the model


# ── Helpers ─────────────────────────────────────────────────────────────────

def _find_roughness_files(directory: Path) -> list[Path]:
    """Return all ``roughness_*.tif`` files under *directory* (recursive)
    that are **not** raw pre-UTM files (``*_raw.tif``)."""
    return sorted(
        p for p in directory.rglob("roughness_*.tif")
        if not p.name.endswith("_raw.tif")
    )


def _load_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Read a single-band raster; nodata pixels become ``np.nan``."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, profile


def _load_metadata(roughness_tif: Path) -> dict:
    """Return the contents of ``terrain_metadata.json`` from the same folder,
    or an empty dict when the file is missing."""
    meta_path = roughness_tif.parent / "terrain_metadata.json"
    if meta_path.exists():
        with meta_path.open() as fh:
            return json.load(fh)
    return {}


def _raw_counterpart(roughness_tif: Path) -> Path | None:
    """Return the ``_raw.tif`` file that corresponds to *roughness_tif*,
    or ``None`` if it does not exist."""
    raw = roughness_tif.parent / (roughness_tif.stem + "_raw.tif")
    return raw if raw.exists() else None


# ── Core analysis ────────────────────────────────────────────────────────────

def _compute_stats(data: np.ndarray, threshold: float) -> dict:
    """Return a dict of summary statistics for *data*."""
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return {
            "n_valid": 0, "n_total": data.size,
            "min": np.nan, "max": np.nan, "mean": np.nan,
            "median": np.nan, "std": np.nan,
            "n_above": 0, "frac_above": 0.0,
        }
    n_above = int(np.sum(valid > threshold))
    return {
        "n_valid":    int(valid.size),
        "n_total":    int(data.size),
        "min":        float(np.min(valid)),
        "max":        float(np.max(valid)),
        "mean":       float(np.mean(valid)),
        "median":     float(np.median(valid)),
        "std":        float(np.std(valid)),
        "n_above":    n_above,
        "frac_above": n_above / valid.size,
    }


def _attribute_sources(
    data: np.ndarray,
    threshold: float,
    tree_fallback_z0: float,
    tolerance: float = 0.01,
) -> dict:
    """Break down pixels above *threshold* into canopy-derived vs fixed fallback.

    Parameters
    ----------
    data:
        z0 array (NaN where nodata).
    threshold:
        Only pixels with ``z0 > threshold`` are considered.
    tree_fallback_z0:
        The fixed z0 value used for tree pixels that had no canopy height data
        (e.g. 1.5 m for GWA4).
    tolerance:
        A pixel is classified as "fixed fallback" when
        ``|z0 − tree_fallback_z0| ≤ tolerance``.  In the UTM-reprojected
        raster bilinear resampling smears exact values, so a small tolerance
        is needed.

    Returns
    -------
    dict with keys ``n_above``, ``n_canopy``, ``n_fixed``, ``frac_canopy``,
    ``frac_fixed``.
    """
    above = data[np.isfinite(data) & (data > threshold)]
    n_above = above.size
    if n_above == 0:
        return {"n_above": 0, "n_canopy": 0, "n_fixed": 0,
                "frac_canopy": 0.0, "frac_fixed": 0.0}

    near_fallback = np.abs(above - tree_fallback_z0) <= tolerance
    # Also treat values at the model cap as canopy-derived (z0 was capped at 3 m)
    # because the cap is only ever reached via the 0.1×h formula (h ≥ 30 m).
    at_cap = np.abs(above - _ORA_Z0_MAX_CAP_M) <= tolerance

    n_fixed   = int(np.sum(near_fallback & ~at_cap))
    n_canopy  = n_above - n_fixed
    return {
        "n_above":     n_above,
        "n_canopy":    n_canopy,
        "n_fixed":     n_fixed,
        "frac_canopy": n_canopy / n_above,
        "frac_fixed":  n_fixed  / n_above,
    }


# ── Visualisation ────────────────────────────────────────────────────────────

def _save_annotated_png(
    data: np.ndarray,
    profile: dict,
    threshold: float,
    output_png: Path,
) -> None:
    """Save a two-panel figure:

    Left  – full roughness map (viridis, log-scaled).
    Right – binary mask of pixels above *threshold* (red) overlaid on the
            roughness map (greyscale).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tf = profile["transform"]

    # ── Left: full roughness map ────────────────────────────────────────────
    ax = axes[0]
    # Use a log normalisation so that the many low-z0 pixels are visible while
    # the high-z0 tail is still represented.
    valid_pos = data[np.isfinite(data) & (data > 0)]
    vmin = float(np.percentile(valid_pos, 1))  if valid_pos.size else 1e-4
    vmax = float(np.nanmax(data))               if valid_pos.size else 1.0
    vmin = max(vmin, 1e-4)

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(
        masked,
        norm=norm,
        cmap="viridis",
        extent=[
            tf.c, tf.c + tf.a * data.shape[1],
            tf.f + tf.e * data.shape[0], tf.f,
        ],
        origin="upper",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("z₀  (m)")
    ax.set_title("Roughness map (z₀)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # ── Right: base roughness (greyscale) + above-threshold overlay ─────────
    ax = axes[1]
    grey_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    ax.imshow(
        masked,
        norm=grey_norm,
        cmap="Greys_r",
        extent=[
            tf.c, tf.c + tf.a * data.shape[1],
            tf.f + tf.e * data.shape[0], tf.f,
        ],
        origin="upper",
    )

    # Build red alpha mask for above-threshold pixels
    mask_above = np.where(np.isfinite(data) & (data > threshold), 0.75, 0.0).astype(np.float32)
    rgba = np.zeros((*data.shape, 4), dtype=np.float32)
    rgba[..., 0] = 1.0   # red channel
    rgba[..., 3] = mask_above
    ax.imshow(
        rgba,
        extent=[
            tf.c, tf.c + tf.a * data.shape[1],
            tf.f + tf.e * data.shape[0], tf.f,
        ],
        origin="upper",
        interpolation="none",
    )
    ax.set_title(f"Regions with z₀ > {threshold} m  (red)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    fig.suptitle(output_png.stem, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_histogram_png(
    data: np.ndarray,
    threshold: float,
    stats: dict,
    output_png: Path,
) -> None:
    """Save a histogram of z0 values with a vertical line at *threshold*."""
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(
        np.log10(max(float(np.min(valid)), 1e-4)),
        np.log10(float(np.max(valid))),
        60,
    )
    ax.hist(valid, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(threshold, color="crimson", linewidth=1.5,
               label=f"threshold = {threshold} m  ({stats['frac_above']:.1%} of pixels)")
    ax.set_xscale("log")
    ax.set_xlabel("z₀  (m)")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"z₀ distribution — {output_png.stem}")
    ax.legend()
    stats_text = (
        f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
        f"mean={stats['mean']:.4f}  median={stats['median']:.4f}"
    )
    ax.annotate(stats_text, xy=(0.02, 0.97), xycoords="axes fraction",
                va="top", fontsize=8, family="monospace")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Per-file entry point ─────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def analyse_file(
    tif_path: Path,
    threshold: float,
    output_dir: Path | None,
) -> None:
    """Run the full analysis pipeline for a single roughness map."""
    _print_section(str(tif_path))

    # ── Load data ────────────────────────────────────────────────────────────
    data, profile = _load_raster(tif_path)
    meta = _load_metadata(tif_path)
    roughness_meta = meta.get("roughness", {})

    # ── Model parameters from metadata ──────────────────────────────────────
    canopy_used      = roughness_meta.get("canopy_height_used", "unknown")
    tree_fallback    = roughness_meta.get("z0_tree_fallback_m", None)
    z0_tree_factor   = roughness_meta.get("z0_tree_factor", _ORA_Z0_TREE_FACTOR)
    z0_cap           = roughness_meta.get("z0_max_cap_m", _ORA_Z0_MAX_CAP_M)
    lc_table         = roughness_meta.get("land_cover_table", "unknown")

    print(f"  Land-cover table  : {lc_table}")
    print(f"  Canopy height used: {canopy_used}")
    if tree_fallback is not None:
        print(f"  Tree z0 fallback  : {tree_fallback} m  "
              f"(used when canopy height unavailable)")
    print(f"  z0 = {z0_tree_factor} × canopy_height  (for tree pixels with data)")
    print(f"  Hard cap          : z0 ≤ {z0_cap} m")

    # ── Statistics ───────────────────────────────────────────────────────────
    stats = _compute_stats(data, threshold)
    print(f"\n  Raster size: {data.shape[1]} × {data.shape[0]} px  "
          f"({stats['n_valid']:,} valid / {stats['n_total']:,} total)")
    print(f"  min    = {stats['min']:.4f} m")
    print(f"  max    = {stats['max']:.4f} m")
    print(f"  mean   = {stats['mean']:.4f} m")
    print(f"  median = {stats['median']:.4f} m")
    print(f"  std    = {stats['std']:.4f} m")
    print(f"\n  Pixels with z0 > {threshold} m : "
          f"{stats['n_above']:,} / {stats['n_valid']:,}  "
          f"({stats['frac_above']:.2%})")

    # ── Attribution analysis ─────────────────────────────────────────────────
    # Prefer the raw (pre-UTM) roughness file for attribution because
    # bilinear reprojection blurs discrete table values; in the raw file
    # fixed-fallback pixels are still exact.
    raw_path = _raw_counterpart(tif_path)
    if raw_path is not None:
        attr_data, _ = _load_raster(raw_path)
        attr_note = f"(using raw pre-UTM file: {raw_path.name})"
    else:
        attr_data = data
        attr_note = "(using UTM file; tolerance=±0.01 m applied)"

    if tree_fallback is not None and stats["n_above"] > 0:
        attr = _attribute_sources(attr_data, threshold, tree_fallback)
        print(f"\n  Source attribution for pixels with z0 > {threshold} m  {attr_note}:")
        print(f"    Canopy-derived z0  : {attr['n_canopy']:,} px  "
              f"({attr['frac_canopy']:.1%})")
        print(f"      → tree pixels where z0 = {z0_tree_factor} × canopy_height "
              f"(h > {threshold / z0_tree_factor:.1f} m to exceed threshold)")
        print(f"    Fixed fallback z0  : {attr['n_fixed']:,} px  "
              f"({attr['frac_fixed']:.1%})")
        print(f"      → tree pixels with no canopy data; "
              f"assigned table fallback z0 = {tree_fallback} m")
        if tree_fallback <= threshold:
            print(f"      (note: fallback {tree_fallback} m ≤ threshold {threshold} m,"
                  " so fixed-fallback pixels do NOT appear above threshold)")
    elif stats["n_above"] > 0:
        print(
            f"\n  Attribution: no tree_fallback value in metadata; "
            "cannot split canopy vs fixed source.\n"
            f"  All z0 > {threshold} m pixels originate from tree cover "
            "(class 10 in WorldCover).\n"
            f"  With the ORA model (z0 = {z0_tree_factor} × h), z0 > {threshold} m "
            f"requires canopy height h > {threshold / z0_tree_factor:.1f} m."
        )

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_dir = output_dir if output_dir is not None else tif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = tif_path.stem
    map_png  = out_dir / f"{stem}_analysis_map.png"
    hist_png = out_dir / f"{stem}_analysis_histogram.png"

    _save_annotated_png(data, profile, threshold, map_png)
    print(f"\n  Saved annotated map : {map_png}")

    _save_histogram_png(data, threshold, stats, hist_png)
    print(f"  Saved histogram     : {hist_png}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse terrain-fetcher roughness maps: statistics, "
            "annotated PNG, and source attribution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="ROUGHNESS.TIF",
        help="One or more roughness GeoTIFF files to analyse.",
    )
    parser.add_argument(
        "--dir",
        metavar="DIRECTORY",
        help=(
            "Scan a directory (recursively) for roughness_*.tif files. "
            "Can be combined with explicit file arguments."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        metavar="Z0_M",
        help="z0 threshold in metres (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help=(
            "Directory where annotated PNGs are written. "
            "Defaults to the same folder as each input file."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    tif_paths: list[Path] = [Path(f) for f in args.files]

    if args.dir:
        tif_paths += _find_roughness_files(Path(args.dir))

    if not tif_paths:
        print(
            "No roughness files found.  "
            "Pass file paths as positional arguments or use --dir.",
            file=sys.stderr,
        )
        return 1

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in tif_paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)

    output_dir = Path(args.output_dir) if args.output_dir else None

    errors = 0
    for tif_path in unique_paths:
        if not tif_path.exists():
            print(f"File not found: {tif_path}", file=sys.stderr)
            errors += 1
            continue
        try:
            analyse_file(tif_path, threshold=args.threshold, output_dir=output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Error analysing {tif_path}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            traceback.print_exc()
            errors += 1

    print(f"\nDone. Analysed {len(unique_paths) - errors} file(s).")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
