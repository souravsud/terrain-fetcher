"""Tests for roughness map preparation.

These tests inspect and validate:
  1. The ESA WorldCover land-use classifications loaded for a given area.
  2. The z0 lookup table (and its source) used to convert those
     classifications into aerodynamic roughness length values.
  3. The code→z0 mapping itself – including the edge-cases that can produce
     "fishy" (NaN) roughness values in the output raster.

Diagnostic plots are written to ``tests/plots/`` (created automatically) every
time the relevant tests run.  View them directly after a test run::

    pytest tests/test_roughness_map.py -v -s
    # → tests/plots/lookup_table_GWA4.png
    # → tests/plots/synthetic_patch.png

For the network-dependent integration plots (real WorldCover tiles)::

    pytest tests/test_roughness_map.py -v -s --integration
    # → tests/plots/real_<label>.png  (one per parametrized location)

"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest
import windkit as wk

# ---------------------------------------------------------------------------
# Reference: ESA WorldCover 2020/2021 class codes and canonical descriptions
# https://esa-worldcover.org/en/data
# ---------------------------------------------------------------------------
ESA_WORLDCOVER_CLASSES: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Official ESA WorldCover per-class colors (from the product specification)
_LC_COLORS: dict[int, str] = {
    10: "#006400",
    20: "#FFBB22",
    30: "#FFFF4C",
    40: "#F096FF",
    50: "#FA0000",
    60: "#B4B4B4",
    70: "#F0F0F0",
    80: "#0064C8",
    90: "#0096A0",
    95: "#00CF75",
    100: "#FAE6A0",
}

# Directory where all diagnostic plots are written
_PLOTS_DIR = Path(__file__).parent / "plots"


# ---------------------------------------------------------------------------
# Helpers (mirror the logic inside download_square_data)
# ---------------------------------------------------------------------------

_TABLE_WIDTH = 65  # character width used by all diagnostic table output


def _divider(char: str = "=") -> str:
    return char * _TABLE_WIDTH


def _build_lc_code_to_z0(table_name: str = "GWA4") -> dict:
    """Return the {landcover_code: z0} dict as built by download_square_data."""
    lct = wk.get_landcover_table(table_name)
    return {
        lc_id: params.get("z0")
        for lc_id, params in lct.items()
        if params is not None and "z0" in params
    }


def _print_breakdown_header(title: str, extra_lines: list[str] | None = None) -> None:
    """Print a standardised section header for diagnostic tables."""
    print(f"\n{_divider()}")
    print(f"  {title}")
    for line in (extra_lines or []):
        print(f"  {line}")
    print(_divider())


def _print_lookup_table(table_name: str = "GWA4") -> None:
    """Pretty-print the complete landcover → z0 lookup table with source info."""
    lct = wk.get_landcover_table(table_name)
    lc_code_to_z0 = _build_lc_code_to_z0(table_name)

    _print_breakdown_header(
        f"Landcover lookup table  :  '{table_name}'",
        extra_lines=[
            f"Source library          :  windkit=={wk.__version__}",
            f"Table function          :  windkit.get_landcover_table('{table_name}')",
            f"Number of entries       :  {len(lct)}",
        ],
    )
    print(f"{'Code':>6}  {'z0 (m)':>8}  {'d (m)':>6}  Description")
    print(_divider("-"))
    for code in sorted(lct.keys()):
        params = lct[code] or {}
        z0 = lc_code_to_z0.get(code, "—")
        d = params.get("d", "—")
        desc = params.get("desc", "—")
        worldcover_label = ESA_WORLDCOVER_CLASSES.get(code, "")
        suffix = f"  [{worldcover_label}]" if worldcover_label else ""
        print(f"{code:>6}  {str(z0):>8}  {str(d):>6}  {desc}{suffix}")
    print()

    missing = [
        f"{code} ({name})"
        for code, name in ESA_WORLDCOVER_CLASSES.items()
        if code not in lc_code_to_z0
    ]
    if missing:
        print(
            f"  WARNING – the following standard ESA WorldCover codes have NO z0\n"
            f"  mapping and will silently become NaN in the roughness raster:\n"
            f"  {', '.join(missing)}"
        )
    else:
        print("  ✓ All standard ESA WorldCover codes are covered.")
    print(f"{_divider()}\n")


# ---------------------------------------------------------------------------
# Plot helpers – all figures saved to tests/plots/ (created automatically)
# ---------------------------------------------------------------------------

def _get_plots_dir() -> Path:
    """Return (and create) the directory where diagnostic plots are saved."""
    _PLOTS_DIR.mkdir(exist_ok=True)
    return _PLOTS_DIR


def _plot_lookup_table_bar_chart(table_name: str = "GWA4") -> None:
    """Save a horizontal bar chart of z0 values for the standard WorldCover
    classes in *table_name*.

    File: ``tests/plots/lookup_table_{table_name}.png``
    """
    plt.switch_backend("Agg")

    lc_code_to_z0 = _build_lc_code_to_z0(table_name)

    codes = sorted(c for c in ESA_WORLDCOVER_CLASSES if c in lc_code_to_z0)
    labels = [f"{c}: {ESA_WORLDCOVER_CLASSES[c]}" for c in codes]
    z0_vals = [lc_code_to_z0[c] or 0.0 for c in codes]
    colors = [_LC_COLORS.get(c, "#888888") for c in codes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, z0_vals, color=colors, edgecolor="0.3", linewidth=0.6)
    ax.set_xlabel("Aerodynamic roughness length  z₀  (m)", fontsize=11)
    ax.set_title(
        f"GWA4 landcover → z₀ lookup table\n"
        f"Source: windkit.get_landcover_table('{table_name}')  "
        f"[windkit=={wk.__version__}]",
        fontsize=10,
    )
    ax.bar_label(bars, fmt="%.4g", padding=4, fontsize=9)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.18)  # room for bar labels
    ax.invert_yaxis()

    fig.tight_layout()
    out_path = _get_plots_dir() / f"lookup_table_{table_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved lookup table plot → {out_path}")


def _plot_landcover_and_roughness(
    lc_data: np.ndarray,
    lc_code_to_z0: dict,
    title_prefix: str,
    out_stem: str,
) -> None:
    """Save a side-by-side figure: landcover classification and roughness map.

    Left panel – categorical landcover map coloured with official ESA WorldCover
    colours; unknown/nodata pixels shown in grey.

    Right panel – continuous roughness length z₀ map (viridis); NaN pixels
    (unmapped codes) highlighted in red with an annotation.

    Parameters
    ----------
    lc_data:
        2-D ``uint8`` array of ESA WorldCover class codes.
    lc_code_to_z0:
        Mapping from class code to z₀ value (from :func:`_build_lc_code_to_z0`).
    title_prefix:
        Short descriptor used in subplot titles.
    out_stem:
        Output filename stem (without extension).  Saved under ``tests/plots/``.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt.switch_backend("Agg")

    z0_data = np.vectorize(lambda c: lc_code_to_z0.get(int(c)))(lc_data).astype(float)
    unique_codes = sorted(int(c) for c in np.unique(lc_data))

    # --- build discrete colormap from official ESA colours ---
    palette = [_LC_COLORS.get(c, "#888888") for c in unique_codes]
    cmap_lc = ListedColormap(palette)
    bounds_lc = [c - 0.5 for c in unique_codes] + [unique_codes[-1] + 0.5]
    norm_lc = BoundaryNorm(bounds_lc, len(unique_codes))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: landcover classification
    axes[0].imshow(lc_data, cmap=cmap_lc, norm=norm_lc, interpolation="nearest")
    axes[0].set_title(f"{title_prefix}\nLandcover classification (ESA WorldCover)")
    axes[0].set_xlabel("pixel column")
    axes[0].set_ylabel("pixel row")
    patches = [
        mpatches.Patch(
            color=palette[i],
            label=f"{c}: {ESA_WORLDCOVER_CLASSES.get(c, 'Unknown/nodata')}",
        )
        for i, c in enumerate(unique_codes)
    ]
    axes[0].legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.85)

    # Panel 2: roughness length z0
    cmap_z0 = plt.cm.viridis.copy()
    cmap_z0.set_bad(color="red", alpha=0.8)  # NaN → red
    im = axes[1].imshow(z0_data, cmap=cmap_z0, interpolation="nearest")
    axes[1].set_title(f"{title_prefix}\nRoughness length  z₀  (m)")
    axes[1].set_xlabel("pixel column")
    axes[1].set_ylabel("pixel row")
    plt.colorbar(im, ax=axes[1], label="z₀ (m)", shrink=0.85)

    nan_count = int(np.isnan(z0_data).sum())
    if nan_count:
        axes[1].text(
            0.02, 0.02,
            f"⚠ {nan_count} NaN pixel(s) – unmapped code",
            transform=axes[1].transAxes,
            fontsize=8, color="red", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

    fig.tight_layout()
    out_path = _get_plots_dir() / f"{out_stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved landcover + roughness plot → {out_path}")


# ---------------------------------------------------------------------------
# Fixture: gate integration tests behind the --integration CLI flag
# (the option itself is declared in conftest.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def integration(request):
    """Skip the test unless --integration was passed."""
    if not request.config.getoption("--integration"):
        pytest.skip("Pass --integration to run network-dependent tests.")


# ===========================================================================
# 1. Lookup table unit tests (no network required)
# ===========================================================================

class TestLandcoverLookupTable:
    """Validate the landcover → z0 lookup table used by the roughness pipeline."""

    def test_table_loads_successfully(self):
        """get_landcover_table('GWA4') returns a non-empty mapping."""
        lct = wk.get_landcover_table("GWA4")
        assert len(lct) > 0, "Landcover table must not be empty"

    def test_all_worldcover_codes_present(self):
        """Every standard ESA WorldCover class code has a z0 entry in GWA4.

        A missing code means those pixels will map to None and end up as NaN
        (or garbage) in the final float32 roughness raster – the most likely
        cause of 'fishy' roughness values.
        """
        lc_code_to_z0 = _build_lc_code_to_z0("GWA4")
        missing = [
            f"{code} ({ESA_WORLDCOVER_CLASSES[code]})"
            for code in ESA_WORLDCOVER_CLASSES
            if code not in lc_code_to_z0
        ]
        assert missing == [], (
            f"ESA WorldCover codes with no z0 mapping in GWA4: {missing}. "
            "Pixels with these codes will silently become NaN in the roughness map."
        )

    def test_z0_values_are_non_negative(self):
        """All z0 values in GWA4 must be ≥ 0."""
        lc_code_to_z0 = _build_lc_code_to_z0("GWA4")
        negative = {k: v for k, v in lc_code_to_z0.items() if v is not None and v < 0}
        assert negative == {}, f"Negative z0 values found: {negative}"

    def test_z0_values_are_physically_plausible(self):
        """z0 values should be within physically observed range (0–5 m)."""
        lc_code_to_z0 = _build_lc_code_to_z0("GWA4")
        out_of_range = {
            k: v
            for k, v in lc_code_to_z0.items()
            if v is not None and v > 5.0
        }
        assert out_of_range == {}, (
            f"Unexpectedly large z0 values (> 5 m): {out_of_range}"
        )

    def test_lookup_table_verbose(self, capsys):
        """Print a full diagnostic of all landcover codes, z0 values, and source.

        Run ``pytest -s`` to see the output.
        """
        _print_lookup_table("GWA4")
        captured = capsys.readouterr()
        assert "windkit" in captured.out
        assert "GWA4" in captured.out

    def test_plot_lookup_table_bar_chart(self):
        """Save a bar-chart of z0 values for all standard WorldCover classes.

        Output: ``tests/plots/lookup_table_GWA4.png``
        Run ``pytest -s`` to see the saved path printed to stdout.
        """
        _plot_lookup_table_bar_chart("GWA4")
        out_path = _PLOTS_DIR / "lookup_table_GWA4.png"
        assert out_path.exists(), f"Expected plot not found at {out_path}"
        assert out_path.stat().st_size > 0, "Plot file is empty"


# ===========================================================================
# 2. Code → z0 mapping tests using synthetic landcover arrays
# ===========================================================================

class TestZ0MappingWithSyntheticData:
    """Test the vectorised code→z0 mapping with controlled landcover arrays."""

    def test_known_codes_map_to_expected_z0(self):
        """Each WorldCover pixel code produces the correct z0 value."""
        lc_code_to_z0 = _build_lc_code_to_z0("GWA4")

        # Build a small array covering several WorldCover classes
        codes = np.array(
            [[10, 30, 50],
             [80, 60, 40],
             [90, 70, 100]],
            dtype=np.uint8,  # WorldCover tiles are uint8
        )

        z0_array = np.vectorize(lc_code_to_z0.get)(codes)

        for row in range(codes.shape[0]):
            for col in range(codes.shape[1]):
                code = int(codes[row, col])
                expected = lc_code_to_z0.get(code)
                actual = z0_array[row, col]
                assert actual == expected, (
                    f"Code {code} ({ESA_WORLDCOVER_CLASSES.get(code, '?')}): "
                    f"expected z0={expected}, got {actual}"
                )

    def test_uint8_keys_resolve_correctly(self):
        """Dict keys are Python ints; numpy uint8 values must still look them up."""
        lc_code_to_z0 = _build_lc_code_to_z0("GWA4")

        # Confirm that a numpy uint8 pixel code finds the Python-int dict key
        pixel_uint8 = np.uint8(50)  # "Built-up"
        result = lc_code_to_z0.get(pixel_uint8)
        assert result is not None, (
            "numpy uint8(50) failed to match the Python int key 50 in the "
            "lookup dict – this would silently zero-out built-up areas."
        )
        assert result == lc_code_to_z0[50]

    def test_unknown_code_maps_to_none(self):
        """A pixel code absent from the lookup returns None."""
        lc_code_to_z0 = _build_lc_code_to_z0("GWA4")
        unknown = 255  # common nodata value in uint8 rasters
        assert unknown not in lc_code_to_z0, (
            "Precondition: choose a code that genuinely isn't in the table"
        )

        arr = np.array([[unknown]], dtype=np.uint8)
        z0_arr = np.vectorize(lc_code_to_z0.get)(arr)
        assert z0_arr[0, 0] is None, (
            "Unknown code should map to None; it will become NaN when cast to float32."
        )

    def test_none_becomes_nan_in_float32(self):
        """None values silently become NaN (not 0) when the array is cast to float32.

        This is almost certainly the source of 'fishy' roughness values: any pixel
        whose WorldCover code is missing from the lookup table ends up as NaN in the
        output raster instead of raising an error.
        """
        lc_code_to_z0 = _build_lc_code_to_z0("GWA4")

        unknown = 255  # nodata / unrecognised code
        codes = np.array([[50, unknown]], dtype=np.uint8)
        z0_raw = np.vectorize(lc_code_to_z0.get)(codes)
        z0_f32 = z0_raw.astype(np.float32)

        assert not np.isnan(z0_f32[0, 0]), (
            "z0 for Built-up (code 50) must not be NaN"
        )
        assert np.isnan(z0_f32[0, 1]), (
            f"Code {unknown} (not in lookup) must produce NaN in the float32 raster. "
            "If real WorldCover tiles contain this code, the roughness map will "
            "have NaN patches – the likely cause of 'fishy' values."
        )

    def test_landcover_breakdown_of_synthetic_patch(self, capsys):
        """Print a per-code breakdown for a synthetic landcover array.

        This mirrors what you would see for a real tile – run with ``pytest -s``
        to inspect the diagnostic output.
        """
        lc_code_to_z0 = _build_lc_code_to_z0("GWA4")

        # Simulate a small patch with realistic WorldCover codes plus a nodata pixel
        lc_data = np.array(
            [[30, 30, 40, 40, 50],
             [30, 10, 10, 40, 50],
             [80, 80, 30, 40, 60],
             [80, 80, 30, 30, 255]],  # 255 = nodata / unknown
            dtype=np.uint8,
        )

        unique_codes, counts = np.unique(lc_data, return_counts=True)
        total = lc_data.size

        _print_breakdown_header(
            "Landcover classification breakdown  (synthetic 4×5 patch)",
            extra_lines=[f"Lookup table used: 'GWA4'  (windkit=={wk.__version__})"],
        )
        print(f"{'Code':>6}  {'Pixels':>7}  {'%':>6}  {'z0 (m)':>8}  Class description")
        print(_divider("-"))
        any_missing = False
        for code, count in zip(unique_codes, counts):
            z0 = lc_code_to_z0.get(int(code))
            desc = ESA_WORLDCOVER_CLASSES.get(int(code), "Unknown / nodata")
            flag = "  ← NOT IN LOOKUP → NaN" if z0 is None else ""
            if z0 is None:
                any_missing = True
            print(
                f"{code:>6}  {count:>7}  {100*count/total:>5.1f}%  "
                f"{str(z0):>8}  {desc}{flag}"
            )
        print()
        if any_missing:
            print(
                "  ⚠  One or more codes are absent from the lookup table.\n"
                "     Those pixels will become NaN in the roughness raster."
            )
        else:
            print("  ✓ All codes in this patch have a valid z0 mapping.")
        print(f"{_divider()}\n")

        captured = capsys.readouterr()
        assert "Landcover classification breakdown" in captured.out
        assert "NOT IN LOOKUP" in captured.out  # 255 is absent from GWA4

        # --- save diagnostic plots ---
        _plot_landcover_and_roughness(
            lc_data, lc_code_to_z0,
            title_prefix="Synthetic 4×5 patch",
            out_stem="synthetic_patch",
        )
        out_path = _PLOTS_DIR / "synthetic_patch.png"
        assert out_path.exists(), f"Expected plot not found at {out_path}"


# ===========================================================================
# 3. Integration test – real WorldCover tile download (requires network)
# ===========================================================================

class TestRoughnessMapRealCoordinates:
    """Integration tests that download real WorldCover data from ESA S3.

    These tests are skipped by default.  Run with ``--integration`` to execute.
    """

    @pytest.mark.parametrize("lat,lon,label", [
        (52.52, 13.40, "Berlin, Germany"),
        (55.68, 12.57, "Copenhagen, Denmark"),
    ])
    def test_landcover_codes_and_z0_for_real_location(
        self, lat, lon, label, integration, tmp_path, capsys
    ):
        """Download WorldCover tiles for *lat*/*lon*, print landcover codes
        found in the raw data and show the z0 lookup for each.

        Pass ``--integration`` to enable this test.
        """
        from terrain_fetcher.download_raster import (
            _WORLDCOVER_GRID_URL,
            _calculate_bounds,
            stitch_tiles,
        )
        import geopandas as gpd
        from shapely.geometry import Polygon

        table_name = "GWA4"
        version = "v100"
        year = 2020
        side_km = 10.0  # small area for fast download

        bounds, corners = _calculate_bounds(side_km, lat, lon)
        grid_url = _WORLDCOVER_GRID_URL.format(version=version, year=year)

        _print_breakdown_header(
            f"Location  : {label}  (lat={lat}, lon={lon})",
            extra_lines=[
                f"Bounds    : {bounds}",
                f"Area      : {side_km} km × {side_km} km",
                f"Source    : ESA WorldCover {year} {version}",
                f"Grid URL  : {grid_url}",
                f"LC table  : windkit.get_landcover_table('{table_name}')",
                f"windkit   : {wk.__version__}",
            ],
        )

        # --- identify tiles ---
        grid = gpd.read_file(grid_url)
        aoi = Polygon([(lon_, lat_) for lat_, lon_ in corners])
        tiles = grid[grid.intersects(aoi)].ll_tile.tolist()
        print(f"\n  WorldCover tiles required: {tiles}")

        # --- download and stitch ---
        data_lc, _profile = stitch_tiles(tiles, version, year, bounds)

        # --- build z0 mapping ---
        lc_code_to_z0 = _build_lc_code_to_z0(table_name)

        # --- analyse the raw classifications ---
        unique_codes, counts = np.unique(data_lc, return_counts=True)
        total = data_lc.size

        print(f"\n  Raw WorldCover classification breakdown  ({data_lc.shape[0]}×{data_lc.shape[1]} pixels)")
        print(f"  {'Code':>6}  {'Pixels':>8}  {'%':>6}  {'z0 (m)':>8}  Class description")
        print("  " + _divider("-"))
        any_missing = False
        for code, count in zip(unique_codes, counts):
            z0 = lc_code_to_z0.get(int(code))
            desc = ESA_WORLDCOVER_CLASSES.get(int(code), "Unknown / nodata")
            flag = "  ← NOT IN LOOKUP → NaN" if z0 is None else ""
            if z0 is None:
                any_missing = True
            print(
                f"  {code:>6}  {count:>8}  {100*count/total:>5.1f}%  "
                f"{str(z0):>8}  {desc}{flag}"
            )

        print()
        if any_missing:
            print(
                "  ⚠  Codes absent from the lookup will become NaN in the "
                "roughness raster – likely source of 'fishy' values."
            )
        else:
            print("  ✓ All codes in this tile have a valid z0 mapping.")
        print(f"{_divider()}\n")

        # --- basic sanity checks ---
        assert data_lc.size > 0, "Downloaded tile must not be empty"
        assert len(unique_codes) > 0, "Tile must contain at least one class code"

        # every code that IS in the tile should have a z0 (warn if not)
        unmapped = [
            int(c) for c in unique_codes if lc_code_to_z0.get(int(c)) is None
        ]
        assert unmapped == [], (
            f"Codes present in the real tile but missing from '{table_name}' "
            f"lookup: {unmapped}. These pixels will become NaN in the roughness "
            f"map – this is likely the cause of the 'fishy' roughness values."
        )

        # --- save diagnostic plots ---
        import re
        out_stem = "real_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        _plot_landcover_and_roughness(
            data_lc, lc_code_to_z0,
            title_prefix=f"{label}  ({side_km} km × {side_km} km)",
            out_stem=out_stem,
        )
        out_path = _PLOTS_DIR / f"{out_stem}.png"
        assert out_path.exists(), f"Expected plot not found at {out_path}"

        captured = capsys.readouterr()
        assert label in captured.out


# ===========================================================================
# 4. Integration test – coordinates from the user's config file
# ===========================================================================

def _load_coordinates_from_config(config_path: str) -> list[tuple[float, float, str]]:
    """Parse a terrain-fetcher YAML config and return ``[(lat, lon, label), …]``.

    Supports both input methods accepted by the main application:

    * ``lat`` + ``lon`` keys  – yields a single entry labelled "Config (lat, lon)"
    * ``csv`` key             – delegates to
      :func:`terrain_fetcher.csv_utils.load_coordinates_from_csv` and labels
      each row "Config row N (lat, lon)"

    Returns an empty list when neither ``lat``/``lon`` nor ``csv`` is present.
    """
    import yaml
    from pathlib import Path as _Path

    cfg_path = _Path(config_path)
    with cfg_path.open() as fh:
        data = yaml.safe_load(fh) or {}

    coords: list[tuple[float, float, str]] = []

    if "csv" in data:
        # Resolve relative CSV path against the config file's directory
        csv_path = _Path(data["csv"])
        if not csv_path.is_absolute():
            csv_path = cfg_path.parent / csv_path

        from terrain_fetcher.csv_utils import load_coordinates_from_csv
        raw = load_coordinates_from_csv(str(csv_path))
        for idx, (lat, lon) in enumerate(raw):
            coords.append((lat, lon, f"Config row {idx + 1}  ({lat}, {lon})"))

    elif "lat" in data and "lon" in data:
        lat, lon = float(data["lat"]), float(data["lon"])
        coords.append((lat, lon, f"Config  ({lat}, {lon})"))

    return coords


class TestRoughnessMapFromConfig:
    """Integration test that validates WorldCover data for the coordinates
    defined in the user's own config file.

    Requires both ``--integration`` (network access) and ``--config PATH``
    (path to a terrain-fetcher YAML config).  The test is silently skipped
    when either flag is absent or when the config contains no coordinates.

    Run::

        pytest tests/test_roughness_map.py --integration --config config.yaml -v -s
    """

    def test_landcover_and_roughness_from_config(self, request, capsys):
        """For each coordinate found in ``--config``, download WorldCover tiles,
        validate the z₀ mapping, and save diagnostic plots to ``tests/plots/``.

        *Skipped* when ``--integration`` is absent, ``--config`` is absent, or
        the config file contains no parseable coordinates.
        """
        if not request.config.getoption("--integration"):
            pytest.skip("Pass --integration to run network-dependent tests.")

        config_path = request.config.getoption("--config")
        if not config_path:
            pytest.skip("Pass --config PATH to run the config-coordinates test.")

        coords = _load_coordinates_from_config(config_path)
        if not coords:
            pytest.skip(
                f"Config file '{config_path}' contains no parseable coordinates "
                "(expected 'lat'+'lon' keys or a 'csv' key)."
            )

        from terrain_fetcher.download_raster import (
            _WORLDCOVER_GRID_URL,
            _calculate_bounds,
            stitch_tiles,
        )
        import geopandas as gpd
        import re
        import yaml
        from pathlib import Path as _Path
        from shapely.geometry import Polygon

        # Read WorldCover settings from the config (fall back to defaults)
        with _Path(config_path).open() as fh:
            cfg_data = yaml.safe_load(fh) or {}
        table_name = cfg_data.get("land_cover_table", "GWA4")
        version = cfg_data.get("worldcover_version", "v100")
        year = int(cfg_data.get("worldcover_year", 2020))
        # Cap side_km at 20 km so integration tests stay fast regardless of
        # the value set in the config (which may be much larger, e.g. 50 km).
        side_km = min(float(cfg_data.get("side_km", 10.0)), 20.0)

        lc_code_to_z0 = _build_lc_code_to_z0(table_name)
        grid_url = _WORLDCOVER_GRID_URL.format(version=version, year=year)
        grid = gpd.read_file(grid_url)

        failures: list[str] = []

        for lat, lon, label in coords:
            bounds, corners = _calculate_bounds(side_km, lat, lon)

            _print_breakdown_header(
                f"Location  : {label}",
                extra_lines=[
                    f"Bounds    : {bounds}",
                    f"Area      : {side_km} km × {side_km} km",
                    f"Source    : ESA WorldCover {year} {version}",
                    f"Grid URL  : {grid_url}",
                    f"LC table  : windkit.get_landcover_table('{table_name}')",
                    f"windkit   : {wk.__version__}",
                ],
            )

            aoi = Polygon([(lon_, lat_) for lat_, lon_ in corners])
            tiles = grid[grid.intersects(aoi)].ll_tile.tolist()
            print(f"\n  WorldCover tiles required: {tiles}")

            data_lc, _profile = stitch_tiles(tiles, version, year, bounds)

            unique_codes, counts = np.unique(data_lc, return_counts=True)
            total = data_lc.size

            print(
                f"\n  Raw WorldCover breakdown  "
                f"({data_lc.shape[0]}×{data_lc.shape[1]} pixels)"
            )
            print(
                f"  {'Code':>6}  {'Pixels':>8}  {'%':>6}  {'z0 (m)':>8}  "
                f"Class description"
            )
            print("  " + _divider("-"))
            any_missing = False
            for code, count in zip(unique_codes, counts):
                z0 = lc_code_to_z0.get(int(code))
                desc = ESA_WORLDCOVER_CLASSES.get(int(code), "Unknown / nodata")
                flag = "  ← NOT IN LOOKUP → NaN" if z0 is None else ""
                if z0 is None:
                    any_missing = True
                print(
                    f"  {code:>6}  {count:>8}  {100*count/total:>5.1f}%  "
                    f"{str(z0):>8}  {desc}{flag}"
                )

            print()
            if any_missing:
                print(
                    "  ⚠  Codes absent from the lookup will become NaN in the "
                    "roughness raster – likely source of 'fishy' values."
                )
            else:
                print("  ✓ All codes in this tile have a valid z0 mapping.")
            print(f"{_divider()}\n")

            # Sanity checks (collect failures instead of stopping at the first)
            if data_lc.size == 0:
                failures.append(f"{label}: downloaded tile is empty")
                continue

            unmapped = [
                int(c) for c in unique_codes if lc_code_to_z0.get(int(c)) is None
            ]
            if unmapped:
                failures.append(
                    f"{label}: codes {unmapped} present in tile but missing from "
                    f"'{table_name}' – those pixels will become NaN in the "
                    f"roughness map."
                )

            # Save diagnostic plots
            out_stem = "config_" + re.sub(
                r"[^a-z0-9]+", "_", label.lower()
            ).strip("_")
            _plot_landcover_and_roughness(
                data_lc, lc_code_to_z0,
                title_prefix=label,
                out_stem=out_stem,
            )

        assert failures == [], "\n".join(failures)

        captured = capsys.readouterr()
        # At least the first coordinate's label must appear in the captured output
        assert coords[0][2] in captured.out
