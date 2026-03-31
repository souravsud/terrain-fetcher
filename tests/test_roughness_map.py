"""Tests for roughness map preparation.

These tests inspect and validate:
  1. The ESA WorldCover land-use classifications loaded for a given area.
  2. The z0 lookup table (and its source) used to convert those
     classifications into aerodynamic roughness length values.
  3. The code→z0 mapping itself – including the edge-cases that can produce
     "fishy" (NaN) roughness values in the output raster.

Quality and mapping tests are parametrized over every supported table
(``GWA4`` via windkit and ``custom`` via ``landcover_roughness.csv``) so that
both tables are held to exactly the same correctness bar automatically.

Roughness computation strategy
-------------------------------
* **GWA4 table**: the ORA (Obstacle Roughness Assessment) bivariate model is
  used for all tests.  Tree pixels (code 10) receive ``z₀ = 0.1 × h`` (capped
  at 3 m) from the ETH Global Canopy Height dataset; other classes use the
  GWA4 lookup value.  Tests that supply synthetic landcover arrays also supply
  a matching synthetic canopy-height array so that all ORA branches are
  exercised (below cap, at cap, above cap, and the zero-height fallback).
* **custom table**: a straight vectorised lookup is used (no ORA) because the
  custom table may define code 10 differently.

Diagnostic plots
-----------------
Plots are written to ``tests/plots/`` (created automatically) every time the
relevant tests run::

    pytest tests/test_roughness_map.py -v -s
    # → tests/plots/lookup_table_GWA4.png
    # → tests/plots/lookup_table_custom.png
    # → tests/plots/synthetic_patch_GWA4.png   (3-panel: LC + canopy + z₀)
    # → tests/plots/synthetic_patch_custom.png  (2-panel: LC + z₀)
    # → tests/plots/canopy_height_vs_z0.png

For the network-dependent integration plots (real WorldCover tiles)::

    pytest tests/test_roughness_map.py -v -s --integration
    # → tests/plots/real_<label>.png  (3-panel when ETH canopy is available)
    # → tests/plots/config_<label>.png  (3-panel for GWA4, 2-panel for custom)

Network-dependent test latency
-------------------------------
``--integration`` tests download data from two external sources:

1. **ESA WorldCover grid index** (``esa_worldcover_{year}_grid.geojson``) –
   a ~4 MB GeoJSON file fetched once per test run to identify which 3°×3°
   WorldCover tiles overlap the requested area.  On a slow connection this
   alone can take 10–30 s.
2. **WorldCover tile(s)** – large Cloud-Optimised GeoTIFFs served from ESA S3;
   only the required spatial window is read via HTTP range requests, but the
   initial connection overhead can still add several seconds per tile.
3. **ETH canopy height tile(s)** – fetched from ETH Zürich libdrive; same
   range-request approach, similar latency.

The delay is therefore **network-speed dependent and not a code issue**.  A
first run on a fast connection typically completes in under a minute; on a
slow or metered connection it may take several minutes.

"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest
import windkit as wk

from terrain_fetcher.lc_table import load_custom_landcover_table

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

#: Path to the default custom lookup file shipped with the repository.
_REPO_CUSTOM_TABLE = Path(__file__).parent.parent / "landcover_roughness.csv"

#: Parametrize params covering every supported table configuration.
#: Used by TestLookupTableQuality and TestZ0MappingWithSyntheticData so that
#: both tables are validated by exactly the same test logic.
_TABLE_PARAMS = [
    pytest.param("GWA4", None, id="GWA4"),
    pytest.param("custom", str(_REPO_CUSTOM_TABLE), id="custom"),
]


# ---------------------------------------------------------------------------
# Helpers (mirror the logic inside download_square_data)
# ---------------------------------------------------------------------------

_TABLE_WIDTH = 65  # character width used by all diagnostic table output


def _divider(char: str = "=") -> str:
    return char * _TABLE_WIDTH


def _build_lc_code_to_z0(table_name: str = "GWA4", custom_path: str | None = None) -> dict:
    """Return the {landcover_code: z0} dict as built by download_square_data.

    When *table_name* is ``"custom"``, *custom_path* must point to a valid
    land-cover CSV/text file; :func:`load_custom_landcover_table` is used to
    parse it.  For all other names the table is fetched from windkit.
    """
    if table_name == "custom":
        lct = load_custom_landcover_table(custom_path)
        return {lc_id: params["z0"] for lc_id, params in lct.items()}
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


def _print_lookup_table(table_name: str = "GWA4", custom_path: str | None = None) -> None:
    """Pretty-print the complete landcover → z0 lookup table with source info.

    Supports both windkit built-in tables (e.g. ``"GWA4"``) and user-supplied
    custom files (``table_name="custom"`` with *custom_path* set).
    """
    if table_name == "custom":
        lct = load_custom_landcover_table(custom_path)
        lc_code_to_z0 = {lc_id: params["z0"] for lc_id, params in lct.items()}
        _print_breakdown_header(
            f"Landcover lookup table  :  '{table_name}'",
            extra_lines=[
                f"Source file             :  {custom_path}",
                f"Number of entries       :  {len(lct)}",
            ],
        )
        print(f"{'Code':>6}  {'z0 (m)':>8}  Description")
        print(_divider("-"))
        for code in sorted(lct.keys()):
            params = lct[code]
            z0 = lc_code_to_z0.get(code, "—")
            desc = params.get("description", "—")
            worldcover_label = ESA_WORLDCOVER_CLASSES.get(code, "")
            suffix = f"  [{worldcover_label}]" if worldcover_label else ""
            print(f"{code:>6}  {str(z0):>8}  {desc}{suffix}")
    else:
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


def _plot_lookup_table_bar_chart(
    table_name: str = "GWA4", custom_path: str | None = None
) -> None:
    """Save a horizontal bar chart of z0 values for the standard WorldCover
    classes in *table_name*.

    Supports both windkit built-in tables and custom file-based tables.

    File: ``tests/plots/lookup_table_{table_name}.png``
    """
    plt.switch_backend("Agg")

    lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)

    codes = sorted(c for c in ESA_WORLDCOVER_CLASSES if c in lc_code_to_z0)
    labels = [f"{c}: {ESA_WORLDCOVER_CLASSES[c]}" for c in codes]
    z0_vals = [lc_code_to_z0[c] or 0.0 for c in codes]
    colors = [_LC_COLORS.get(c, "#888888") for c in codes]

    if table_name == "custom":
        title = (
            f"Custom landcover → z₀ lookup table\n"
            f"Source: {custom_path}"
        )
    else:
        title = (
            f"GWA4 landcover → z₀ lookup table\n"
            f"Source: windkit.get_landcover_table('{table_name}')  "
            f"[windkit=={wk.__version__}]"
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, z0_vals, color=colors, edgecolor="0.3", linewidth=0.6)
    ax.set_xlabel("Aerodynamic roughness length  z₀  (m)", fontsize=11)
    ax.set_title(title, fontsize=10)
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
    z0_data: np.ndarray,
    title_prefix: str,
    out_stem: str,
    canopy_data: np.ndarray | None = None,
    want_canopy_panel: bool = False,
) -> None:
    """Save a diagnostic figure combining landcover, optional canopy height,
    and the final roughness length z₀.

    Panel layout
    ------------
    * **2 panels** (default, ``want_canopy_panel=False``): landcover + z₀.
    * **3 panels** when canopy data is requested (``want_canopy_panel=True``):
      landcover + canopy height + z₀.  If *canopy_data* is ``None`` the centre
      panel shows a grey placeholder labelled "ETH canopy data unavailable",
      which tells the user that the download failed rather than silently hiding
      the panel.

    Parameters
    ----------
    lc_data:
        2-D ``uint8`` array of ESA WorldCover class codes.
    z0_data:
        Pre-computed 2-D float32 roughness-length array aligned to *lc_data*.
        Produced by :func:`_compute_ora_z0_d` (GWA4) or a vectorised lookup
        (custom tables).
    title_prefix:
        Short descriptor used in subplot titles.
    out_stem:
        Output filename stem (without extension).  Saved under ``tests/plots/``.
    canopy_data:
        Optional 2-D float32 canopy-height array aligned to *lc_data*.
        When given a third panel shows actual tree heights.
    want_canopy_panel:
        When ``True`` the canopy height panel is always included (with a grey
        placeholder when *canopy_data* is ``None``).  Use this for GWA4/ORA
        tests so the user can see whether the ETH download succeeded.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt.switch_backend("Agg")

    z0_float = np.asarray(z0_data, dtype=float)
    unique_codes = sorted(int(c) for c in np.unique(lc_data))

    # --- build discrete colormap from official ESA colours ---
    palette = [_LC_COLORS.get(c, "#888888") for c in unique_codes]
    cmap_lc = ListedColormap(palette)
    bounds_lc = [c - 0.5 for c in unique_codes] + [unique_codes[-1] + 0.5]
    norm_lc = BoundaryNorm(bounds_lc, len(unique_codes))

    show_canopy = want_canopy_panel or (canopy_data is not None)
    n_panels = 3 if show_canopy else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    axes = list(axes)  # always a list so indexing is uniform

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

    ax_idx = 1

    # Panel 2 (optional): canopy height
    if show_canopy:
        if canopy_data is not None:
            h_display = np.where(
                (canopy_data > 0) & np.isfinite(canopy_data), canopy_data, np.nan
            )
            cmap_h = plt.cm.YlGn.copy()
            cmap_h.set_bad(color="#dddddd")
            im_h = axes[ax_idx].imshow(
                h_display, cmap=cmap_h, interpolation="nearest", vmin=0
            )
            axes[ax_idx].set_title(
                f"{title_prefix}\nCanopy height  h  (m)\n(grey = non-tree / no data)"
            )
            plt.colorbar(im_h, ax=axes[ax_idx], label="h (m)", shrink=0.85)
        else:
            # Grey placeholder when ETH download failed
            axes[ax_idx].set_facecolor("#cccccc")
            axes[ax_idx].text(
                0.5, 0.5,
                "ETH canopy data\nunavailable\n(download failed)",
                ha="center", va="center", fontsize=12, color="#555555",
                transform=axes[ax_idx].transAxes,
            )
            axes[ax_idx].set_title(
                f"{title_prefix}\nCanopy height  h  (m)"
            )
            axes[ax_idx].set_xticks([])  # no pixel axes on a blank placeholder
            axes[ax_idx].set_yticks([])
        axes[ax_idx].set_xlabel("pixel column")
        axes[ax_idx].set_ylabel("pixel row")
        ax_idx += 1

    # Last panel: roughness length z0
    cmap_z0 = plt.cm.viridis.copy()
    cmap_z0.set_bad(color="red", alpha=0.8)  # NaN → red
    im = axes[ax_idx].imshow(z0_float, cmap=cmap_z0, interpolation="nearest")
    axes[ax_idx].set_title(f"{title_prefix}\nRoughness length  z₀  (m)")
    axes[ax_idx].set_xlabel("pixel column")
    axes[ax_idx].set_ylabel("pixel row")
    plt.colorbar(im, ax=axes[ax_idx], label="z₀ (m)", shrink=0.85)

    nan_count = int(np.isnan(z0_float).sum())
    if nan_count:
        axes[ax_idx].text(
            0.02, 0.02,
            f"⚠ {nan_count} NaN pixel(s) – unmapped code",
            transform=axes[ax_idx].transAxes,
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
# 1. Lookup table quality tests (parametrized over all supported tables)
# ===========================================================================

class TestLookupTableQuality:
    """Quality bar for every supported lookup table.

    The same six checks run against both ``GWA4`` (from windkit) and the
    ``custom`` table (``landcover_roughness.csv``) so neither table can regress
    without the tests catching it.
    """

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_table_loads_successfully(self, table_name, custom_path):
        """Table returns a non-empty mapping."""
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)
        assert len(lc_code_to_z0) > 0, f"Lookup table '{table_name}' must not be empty"

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_all_worldcover_codes_present(self, table_name, custom_path):
        """Every standard ESA WorldCover class code has a z0 entry.

        A missing code means those pixels will map to None and end up as NaN
        (or garbage) in the final float32 roughness raster – the most likely
        cause of 'fishy' roughness values.
        """
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)
        missing = [
            f"{code} ({ESA_WORLDCOVER_CLASSES[code]})"
            for code in ESA_WORLDCOVER_CLASSES
            if code not in lc_code_to_z0
        ]
        assert missing == [], (
            f"ESA WorldCover codes with no z0 mapping in '{table_name}': {missing}. "
            "Pixels with these codes will silently become NaN in the roughness map."
        )

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_z0_values_are_non_negative(self, table_name, custom_path):
        """All z0 values must be ≥ 0."""
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)
        negative = {k: v for k, v in lc_code_to_z0.items() if v is not None and v < 0}
        assert negative == {}, f"Negative z0 values found in '{table_name}': {negative}"

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_z0_values_are_physically_plausible(self, table_name, custom_path):
        """z0 values should be within physically observed range (0–5 m)."""
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)
        out_of_range = {
            k: v for k, v in lc_code_to_z0.items() if v is not None and v > 5.0
        }
        assert out_of_range == {}, (
            f"Unexpectedly large z0 values (> 5 m) in '{table_name}': {out_of_range}"
        )

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_verbose_print_mentions_source(self, table_name, custom_path, capsys):
        """Diagnostic print output identifies the table name and its source.

        Run ``pytest -s`` to see the output.
        """
        _print_lookup_table(table_name, custom_path=custom_path)
        captured = capsys.readouterr()
        assert table_name in captured.out
        # GWA4 comes from windkit; custom comes from the file
        expected_source = "windkit" if table_name != "custom" else Path(custom_path).name
        assert expected_source in captured.out

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_plot_bar_chart(self, table_name, custom_path):
        """Save a bar-chart of z0 values for all standard WorldCover classes.

        Output: ``tests/plots/lookup_table_{table_name}.png``
        Run ``pytest -s`` to see the saved path printed to stdout.
        """
        _plot_lookup_table_bar_chart(table_name, custom_path=custom_path)
        out_path = _PLOTS_DIR / f"lookup_table_{table_name}.png"
        assert out_path.exists(), f"Expected plot not found at {out_path}"
        assert out_path.stat().st_size > 0, "Plot file is empty"


# ===========================================================================
# 2. Custom-table-specific tests (no network required)
# ===========================================================================

class TestCustomLandcoverTableOnly:
    """Tests that are specific to the custom file-based table only.

    These cover properties that have no equivalent for windkit built-in tables:
    the CSV file must exist on disk and must include a fallback entry for
    unrecognised class codes.
    """

    def test_custom_table_file_exists(self):
        """landcover_roughness.csv must be present in the repository root."""
        assert _REPO_CUSTOM_TABLE.exists(), (
            f"Default custom table not found at {_REPO_CUSTOM_TABLE}"
        )

    def test_custom_fallback_code_present(self):
        """Class 999 (Unknown / unclassified) must exist as a fallback entry."""
        table = load_custom_landcover_table(_REPO_CUSTOM_TABLE)
        assert 999 in table, (
            "Custom table must include class 999 as the fallback for unrecognised codes."
        )


# ===========================================================================
# 3. Code → z0 mapping tests using synthetic landcover arrays
# ===========================================================================

class TestZ0MappingWithSyntheticData:
    """Test the vectorised code→z0 mapping with controlled landcover arrays.

    All tests are parametrized over both supported tables so that the mapping
    mechanics are validated regardless of which table is used.
    """

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_known_codes_map_to_expected_z0(self, table_name, custom_path):
        """Each WorldCover pixel code produces the correct z0 value."""
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)

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

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_uint8_keys_resolve_correctly(self, table_name, custom_path):
        """Dict keys are Python ints; numpy uint8 values must still look them up."""
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)

        # Confirm that a numpy uint8 pixel code finds the Python-int dict key
        pixel_uint8 = np.uint8(50)  # "Built-up" – present in every supported table
        result = lc_code_to_z0.get(pixel_uint8)
        assert result is not None, (
            "numpy uint8(50) failed to match the Python int key 50 in the "
            "lookup dict – this would silently zero-out built-up areas."
        )
        assert result == lc_code_to_z0[50]

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_unknown_code_maps_to_none(self, table_name, custom_path):
        """A pixel code absent from the lookup returns None."""
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)
        unknown = 255  # common nodata value in uint8 rasters; absent from all tables
        assert unknown not in lc_code_to_z0, (
            "Precondition: choose a code that genuinely isn't in the table"
        )

        arr = np.array([[unknown]], dtype=np.uint8)
        z0_arr = np.vectorize(lc_code_to_z0.get)(arr)
        assert z0_arr[0, 0] is None, (
            "Unknown code should map to None; it will become NaN when cast to float32."
        )

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_none_becomes_nan_in_float32(self, table_name, custom_path):
        """None values silently become NaN (not 0) when the array is cast to float32.

        This is almost certainly the source of 'fishy' roughness values: any pixel
        whose WorldCover code is missing from the lookup table ends up as NaN in the
        output raster instead of raising an error.
        """
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)

        unknown = 255  # nodata / unrecognised code; absent from all tables
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

    @pytest.mark.parametrize("table_name, custom_path", _TABLE_PARAMS)
    def test_landcover_breakdown_of_synthetic_patch(
        self, table_name, custom_path, capsys
    ):
        """Print a per-code breakdown for a synthetic landcover array.

        For **GWA4** the ORA bivariate model is applied via
        :func:`~terrain_fetcher.download_raster._compute_ora_z0_d` with a
        paired canopy-height array.  Tree pixels at heights 5, 12, 25, 30, 40,
        45 m (and zero-height fallback pixels) are included so that all
        branches of the ORA decision tree are exercised in a single test run.
        A **three-panel** diagnostic figure is saved
        (landcover + canopy height + roughness map).

        For **custom** tables a straight vectorised lookup is used (no ORA,
        because the custom table may define code 10 differently).  A
        **two-panel** figure is saved (landcover + roughness map).
        """
        from terrain_fetcher.download_raster import _compute_ora_z0_d

        if table_name == "custom":
            lct = load_custom_landcover_table(custom_path)
            lc_code_to_z0 = {lc_id: params["z0"] for lc_id, params in lct.items()}
        else:
            lct = wk.get_landcover_table(table_name)
            lc_code_to_z0 = _build_lc_code_to_z0(table_name)

        # Synthetic landcover patch (5 rows × 6 cols).
        # Code 10 (Tree cover) appears in a 3×3 block so that different canopy
        # heights can be assigned to adjacent tree pixels.
        lc_data = np.array(
            [[30,  30,  40,  40, 50, 50],
             [30,  10,  10,  10, 50, 20],
             [80,  10,  10,  10, 60, 20],
             [80,  10,  10,  10, 60, 20],
             [80,  80,  30,  30, 30, 255]],  # 255 = nodata; absent from all tables
            dtype=np.uint8,
        )

        # Canopy heights (m) for each pixel.
        # Non-tree pixels carry 0 (irrelevant; ORA only acts on code-10 pixels).
        # Tree-pixel heights cover every ORA branch (rows 1–3, cols 1–3):
        #   col 1 │ col 2 │ col 3 │ ORA outcome
        #  ───────┼───────┼───────┼──────────────────────────────────────────
        #    5 m  │  12 m │  25 m │ z0 = 0.1×h  (below 3.0 m cap)
        #   30 m  │  40 m │  45 m │ z0 = 3.0 m  (at / above cap)
        #    0 m  │   0 m │   0 m │ z0 = table fallback (GWA4: 1.5 m)
        h_data = np.array(
            [[ 0,    0,    0,    0,   0,  0],
             [ 0,    5.0, 12.0, 25.0, 0,  0],   # below cap
             [ 0,   30.0, 40.0, 45.0, 0,  0],   # at/above cap
             [ 0,    0.0,  0.0,  0.0, 0,  0],   # zero → fallback
             [ 0,    0,    0,    0,   0,  0]],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------ #
        # Compute z0                                                           #
        # ------------------------------------------------------------------ #
        if table_name == "GWA4":
            # ORA bivariate model: tree heights drive z0 for code-10 pixels
            z0_data, _ = _compute_ora_z0_d(lc_data, h_data, lct)
            canopy_vis = h_data  # third panel in the diagnostic figure
        else:
            # Non-GWA4 (custom) tables: straight lookup-table mapping
            z0_data = np.vectorize(
                lambda c: float(lc_code_to_z0.get(int(c), np.nan))
            )(lc_data).astype(np.float32)
            canopy_vis = None

        # ------------------------------------------------------------------ #
        # Print per-code breakdown                                             #
        # ------------------------------------------------------------------ #
        unique_codes, counts = np.unique(lc_data, return_counts=True)
        total = lc_data.size

        if table_name == "custom":
            table_info = f"custom file: {Path(custom_path).name}"
        else:
            table_info = f"windkit=={wk.__version__}"
        _print_breakdown_header(
            "Landcover classification breakdown  (synthetic 5×6 patch)",
            extra_lines=[f"Lookup table used: '{table_name}'  ({table_info})"],
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

        # Extra: show ORA results per tree pixel (GWA4 only)
        if table_name == "GWA4":
            print("  ORA model results for tree pixels (code 10):")
            for r, c in np.argwhere(lc_data == 10):
                h_val = float(h_data[r, c])
                z0_val = float(z0_data[r, c])
                if h_val == 0:
                    note = "  (h=0 → fallback)"
                elif h_val * 0.1 >= 3.0:
                    note = "  (capped at 3.0 m)"
                else:
                    note = ""
                print(
                    f"    pixel ({r},{c}): h={h_val:5.1f} m  →  z₀={z0_val:.4f} m{note}"
                )
            print()

        captured = capsys.readouterr()
        assert "Landcover classification breakdown" in captured.out
        assert "NOT IN LOOKUP" in captured.out  # 255 is absent from every table

        # --- ORA model assertions (GWA4 only) ---
        # Expected z0 is derived directly from the ORA formula so that the
        # assertions stay in sync with the h_data values automatically.
        if table_name == "GWA4":
            tree_fallback = float(
                {lc_id: float(params.get("z0", 0.0)) for lc_id, params in lct.items()
                 if params is not None}.get(10, 1.5)
            )
            for (r, c), h_val in [
                ((1, 1),  5.0), ((1, 2), 12.0), ((1, 3), 25.0),  # below cap
                ((2, 1), 30.0), ((2, 2), 40.0), ((2, 3), 45.0),  # at / above cap
            ]:
                expected = min(0.1 * h_val, 3.0)
                assert float(z0_data[r, c]) == pytest.approx(expected, abs=1e-5), (
                    f"pixel ({r},{c}) h={h_val} m: expected z0={expected:.4f}"
                )
            # Zero-height tree pixels → GWA4 table fallback value
            for c in (1, 2, 3):
                assert float(z0_data[3, c]) == pytest.approx(tree_fallback, abs=1e-5), (
                    f"pixel (3,{c}) h=0: expected fallback z0={tree_fallback}"
                )

        # --- save diagnostic plots ---
        _plot_landcover_and_roughness(
            lc_data, z0_data,
            title_prefix=f"Synthetic 5×6 patch ({table_name})",
            out_stem=f"synthetic_patch_{table_name}",
            canopy_data=canopy_vis,
            want_canopy_panel=(table_name == "GWA4"),
        )
        out_path = _PLOTS_DIR / f"synthetic_patch_{table_name}.png"
        assert out_path.exists(), f"Expected plot not found at {out_path}"


# ===========================================================================
# 4. GWA4 bivariate ORA model tests (no network required)
# ===========================================================================

class TestGWA4CanopyBivariate:
    """Unit tests for the direct ORA reclassification logic.

    These tests exercise :func:`terrain_fetcher.download_raster._compute_ora_z0_d`
    with synthetic landcover and canopy-height arrays, covering all branches of
    the bivariate decision tree:

    * Tree pixel + valid height  → ORA formula
    * Tree pixel + zero height   → GWA4 fallback (z0=1.5, d=0)
    * Tree pixel + NaN height    → GWA4 fallback (z0=1.5, d=0)
    * Non-tree pixel             → lookup-table value
    * Height above cap           → z0 clipped to 3.0
    * Height above cap           → d clipped to 25.0
    """

    @pytest.fixture(autouse=True)
    def _import_fn(self):
        from terrain_fetcher.download_raster import _compute_ora_z0_d
        self._fn = _compute_ora_z0_d

    @pytest.fixture()
    def gwa4_lct(self):
        return wk.get_landcover_table("GWA4")

    def _single(self, lc_code: int, h_val, lct):
        """Run on a single pixel and return (z0, d) scalars."""
        lc = np.array([[lc_code]], dtype=np.uint8)
        h = None if h_val is None else np.array([[h_val]], dtype=np.float32)
        z0_arr, d_arr = self._fn(lc, h, lct)
        return float(z0_arr[0, 0]), float(d_arr[0, 0])

    def test_tree_with_valid_height(self, gwa4_lct):
        """Tree pixel (code 10) with h=15 m → ORA: z0=0.1×15=1.5, d=(2/3)×15=10.0."""
        z0, d = self._single(10, 15.0, gwa4_lct)
        assert z0 == pytest.approx(1.5, abs=1e-5), f"Expected z0=1.5, got {z0}"
        assert d == pytest.approx(10.0, abs=1e-5), f"Expected d=10.0, got {d}"

    def test_tree_with_zero_height(self, gwa4_lct):
        """Tree pixel (code 10) with h=0 → fallback: z0=1.5, d=0.0."""
        z0, d = self._single(10, 0.0, gwa4_lct)
        assert z0 == pytest.approx(1.5, abs=1e-5), f"Expected fallback z0=1.5, got {z0}"
        assert d == pytest.approx(0.0), f"Expected d=0.0, got {d}"

    def test_tree_with_nan_height(self, gwa4_lct):
        """Tree pixel (code 10) with h=NaN → fallback: z0=1.5, d=0.0."""
        z0, d = self._single(10, float("nan"), gwa4_lct)
        assert z0 == pytest.approx(1.5, abs=1e-5), f"Expected fallback z0=1.5, got {z0}"
        assert d == pytest.approx(0.0), f"Expected d=0.0, got {d}"

    def test_non_tree_pixel(self, gwa4_lct):
        """Grassland pixel (code 30) → lookup: z0=0.03, d=0.0."""
        z0, d = self._single(30, None, gwa4_lct)
        assert z0 == pytest.approx(0.03, abs=1e-6), f"Expected z0=0.03, got {z0}"
        assert d == pytest.approx(0.0), f"Expected d=0.0, got {d}"

    def test_z0_physical_cap(self, gwa4_lct):
        """Very tall tree (h=40 m) → z0 clipped to 3.0 m."""
        z0, d = self._single(10, 40.0, gwa4_lct)
        assert z0 == pytest.approx(3.0), f"Expected z0 clipped to 3.0, got {z0}"

    def test_d_physical_cap(self, gwa4_lct):
        """Very tall tree (h=50 m) → d clipped to 25.0 m."""
        z0, d = self._single(10, 50.0, gwa4_lct)
        assert d == pytest.approx(25.0), f"Expected d clipped to 25.0, got {d}"


# ===========================================================================
# 5. Visual test – canopy height vs roughness length (ORA curve)
# ===========================================================================

def _plot_canopy_height_vs_z0(lct: dict) -> Path:
    """Save a plot of tree height vs z₀ as predicted by the ORA model.

    Shows:
    * ORA curve  (z0 = 0.1 × h, pre-cap)
    * Effective z₀ after the 3 m physical cap
    * GWA4 fallback value for h = 0 / NaN tree pixels
    * Horizontal reference lines for every non-tree WorldCover class in *lct*
    * Second y-axis showing displacement height d = (2/3) × h (also capped at 25 m)

    File: ``tests/plots/canopy_height_vs_z0.png``
    """
    plt.switch_backend("Agg")

    # Height range (0–50 m in 0.1 m steps)
    h = np.linspace(0.0, 50.0, 500)

    # ORA model: z0 = 0.1 × h  (uncapped, then capped at 3.0 m)
    z0_uncapped = 0.1 * h
    z0_effective = np.clip(z0_uncapped, 0.0, 3.0)

    # Displacement height: d = (2/3) × h, capped at 25 m
    d_effective = np.clip((2.0 / 3.0) * h, 0.0, 25.0)

    # Fallback z0 for tree pixels without valid height (GWA4 class 10)
    lc_code_to_z0 = {
        lc_id: float(params.get("z0", 0.0))
        for lc_id, params in lct.items()
        if params is not None
    }
    tree_fallback_z0 = lc_code_to_z0.get(10, 1.5)

    fig, ax1 = plt.subplots(figsize=(11, 7))

    # --- ORA curve (effective, post-cap) ---
    ax1.plot(h, z0_effective, color="#1a6e2e", linewidth=2.5,
             label="ORA z₀ = 0.1 × h  (capped at 3.0 m)")

    # Show the uncapped portion as a dashed extension
    ax1.plot(h, z0_uncapped, color="#1a6e2e", linewidth=1.2, linestyle="--",
             alpha=0.4, label="ORA z₀ = 0.1 × h  (uncapped)")

    # --- Physical cap annotation ---
    cap_h = 3.0 / 0.1  # h at which cap bites (30 m)
    ax1.axhline(3.0, color="#c0392b", linewidth=1.2, linestyle="-.",
                label="Physical cap  z₀ = 3.0 m")
    ax1.axvline(cap_h, color="#c0392b", linewidth=0.8, linestyle=":",
                alpha=0.5)
    ax1.annotate(
        f"cap bites at h = {cap_h:.0f} m",
        xy=(cap_h, 3.0), xytext=(cap_h + 1.5, 2.7),
        fontsize=8, color="#c0392b",
        arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8),
    )

    # --- GWA4 fallback line (h = 0 / NaN tree pixels) ---
    ax1.axhline(tree_fallback_z0, color="#8e44ad", linewidth=1.2, linestyle="--",
                label=f"Tree fallback (h=0/NaN)  z₀ = {tree_fallback_z0:.2f} m")

    # --- Non-tree class reference lines ---
    non_tree_classes = {
        code: (ESA_WORLDCOVER_CLASSES[code], lc_code_to_z0[code], _LC_COLORS.get(code, "#888"))
        for code in sorted(ESA_WORLDCOVER_CLASSES)
        if code != 10 and code in lc_code_to_z0
    }
    for code, (name, z0_ref, color) in non_tree_classes.items():
        ax1.axhline(z0_ref, color=color, linewidth=0.9, linestyle=":",
                    alpha=0.85, label=f"LC {code}: {name}  (z₀={z0_ref:.4g} m)")

    ax1.set_xlabel("Canopy height  h  (m)", fontsize=12)
    ax1.set_ylabel("Aerodynamic roughness length  z₀  (m)", fontsize=12)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(bottom=0)

    # --- Second y-axis: displacement height d ---
    ax2 = ax1.twinx()
    ax2.plot(h, d_effective, color="#2980b9", linewidth=1.8, linestyle="-",
             alpha=0.7, label="Displacement height  d = (2/3) × h  (capped at 25 m)")
    ax2.set_ylabel("Displacement height  d  (m)", fontsize=12, color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim(bottom=0)

    # --- Combined legend from both axes ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=8, framealpha=0.9,
               title="ORA model  (z₀ = 0.1 × h,  d = (2/3) × h)", title_fontsize=9)

    ax1.set_title(
        "Tree canopy height  →  aerodynamic roughness length & displacement height\n"
        "ORA model · GWA4 lookup table · physical caps: z₀ ≤ 3.0 m, d ≤ 25.0 m",
        fontsize=11,
    )
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = _get_plots_dir() / "canopy_height_vs_z0.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


class TestCanopyHeightVsRoughnessLength:
    """Visual test: save a plot of tree height vs z₀ via the ORA model.

    Run with ``-s`` to see the saved file path::

        pytest tests/test_roughness_map.py::TestCanopyHeightVsRoughnessLength -v -s
        # → tests/plots/canopy_height_vs_z0.png

    The plot shows:

    * **ORA curve** – ``z₀ = 0.1 × h`` (continuous line) and the uncapped
      formula (dashed), so the effect of the 3 m hard cap is immediately visible.
    * **Physical cap** at ``z₀ = 3.0 m`` (bites at h = 30 m).
    * **GWA4 fallback** value used when a tree pixel has h = 0 or NaN.
    * **Non-tree reference lines** – one horizontal line per non-tree WorldCover
      class, coloured with the official ESA colour, so you can compare the ORA
      output against fixed-table values at a glance.
    * **Displacement height** ``d = (2/3) × h`` (capped at 25 m) on a second
      y-axis for a direct side-by-side comparison.
    """

    @pytest.fixture()
    def gwa4_lct(self):
        return wk.get_landcover_table("GWA4")

    def test_plot_canopy_height_vs_z0(self, gwa4_lct):
        """Generate and save the canopy-height vs z₀ ORA plot.

        Assertions:
        * The plot file is created on disk.
        * The ORA curve values match the formula at several spot-check heights.
        * The physical cap is applied correctly.
        * The fallback value equals GWA4 class-10 z₀.
        """
        out_path = _plot_canopy_height_vs_z0(gwa4_lct)
        print(f"\n  Saved canopy height vs z₀ plot → {out_path}")

        assert out_path.exists(), f"Plot file was not created: {out_path}"
        assert out_path.stat().st_size > 10_000, "Plot file looks suspiciously small"

        from terrain_fetcher.download_raster import _compute_ora_z0_d

        def _single(lc_code, h_val):
            lc = np.array([[lc_code]], dtype=np.uint8)
            h = np.array([[h_val]], dtype=np.float32)
            z0_arr, d_arr = _compute_ora_z0_d(lc, h, gwa4_lct)
            return float(z0_arr[0, 0]), float(d_arr[0, 0])

        # Spot-check the ORA formula at a few heights below the cap
        for h_val, expected_z0 in [(5.0, 0.5), (10.0, 1.0), (20.0, 2.0)]:
            z0, _ = _single(10, h_val)
            assert z0 == pytest.approx(expected_z0, abs=1e-6)

        # Physical cap kicks in at h = 30 m (0.1 × 30 = 3.0) and beyond
        z0_at_30, _ = _single(10, 30.0)
        assert z0_at_30 == pytest.approx(3.0)
        z0_at_40, _ = _single(10, 40.0)
        assert z0_at_40 == pytest.approx(3.0)

        # Fallback equals GWA4 class-10 z0
        lc_code_to_z0 = {
            lc_id: float(params.get("z0", 0.0))
            for lc_id, params in gwa4_lct.items()
            if params is not None
        }
        assert lc_code_to_z0[10] == pytest.approx(1.5, abs=1e-5)


# ===========================================================================
# 6. Canopy tile-name logic (offline) and canopy download (integration)
# ===========================================================================

class TestCanopyTileNames:
    """Offline unit tests for :func:`_canopy_tile_names`.

    These tests verify that the 3°-snapped tile-name generator produces the
    correct ETH filename stem for a given bounding box.  No network access is
    required.
    """

    def test_berlin(self):
        """Berlin (lat≈52.5, lon≈13.4) falls inside the N51E012 tile."""
        from terrain_fetcher.download_raster import _canopy_tile_names
        tiles = _canopy_tile_names([13.33, 52.48, 13.47, 52.57])
        assert len(tiles) == 1, f"Expected 1 tile, got {tiles}"
        assert tiles == ["N51E012"]

    def test_portugal(self):
        """Central Portugal (lat≈39.7, lon≈-7.7) falls inside the N39W009 tile."""
        from terrain_fetcher.download_raster import _canopy_tile_names
        tiles = _canopy_tile_names([-7.79, 39.67, -7.67, 39.76])
        assert len(tiles) == 1, f"Expected 1 tile, got {tiles}"
        assert tiles == ["N39W009"]

    def test_southern_hemisphere(self):
        """Southern-hemisphere tile name uses 'S' prefix."""
        from terrain_fetcher.download_raster import _canopy_tile_names
        # -3.5°, -60.5° → SW corner is S06W063
        tiles = _canopy_tile_names([-60.5, -3.5, -60.4, -3.4])
        assert len(tiles) == 1, f"Expected 1 tile, got {tiles}"
        assert tiles == ["S06W063"]

    def test_cross_3deg_lon_boundary(self):
        """Bounds crossing a 3° longitude boundary produce two tile names."""
        from terrain_fetcher.download_raster import _canopy_tile_names
        # lon crosses 12° (a 3° multiple): tiles N51E009 and N51E012
        tiles = _canopy_tile_names([11.9, 51.0, 12.1, 51.5])
        assert set(tiles) == {"N51E009", "N51E012"}

    def test_cross_3deg_lat_boundary(self):
        """Bounds crossing a 3° latitude boundary produce two tile names."""
        from terrain_fetcher.download_raster import _canopy_tile_names
        # lat crosses 51° (a 3° multiple): tiles N48E012 and N51E012
        tiles = _canopy_tile_names([12.0, 50.9, 12.5, 51.1])
        assert set(tiles) == {"N48E012", "N51E012"}


class TestCanopyDownloadIntegration:
    """Integration test for :func:`stitch_canopy_tiles`.

    Verifies that the ETH libdrive COG can actually be opened and a spatial
    window read via GDAL vsicurl.  Requires ``--integration`` (network access).

    Run::

        pytest tests/test_roughness_map.py::TestCanopyDownloadIntegration -v -s --integration
    """

    def test_stitch_canopy_tiles_returns_float32_array(self, integration):
        """stitch_canopy_tiles returns a float32 array for a known-coverage area.

        Uses a small 6 km × 6 km window over the Spessart forest near Würzburg,
        Germany.  This area falls cleanly inside the N48E009 ETH tile and has
        substantial tree cover, so at least some valid (non-NaN) canopy heights
        are expected.

        Assertions
        ----------
        * The function does **not** return ``(None, None)`` – i.e. the COG was
          successfully opened via vsicurl and the window read succeeded.
        * The returned array is 2-D float32.
        * At least one pixel has a finite, positive canopy height.
        * The profile dict contains the expected rasterio keys.
        """
        from terrain_fetcher.download_raster import stitch_canopy_tiles

        # Spessart forest, Bavaria – heavy coniferous cover, tile N48E009
        bounds = [9.90, 49.80, 9.96, 49.86]

        data, prof = stitch_canopy_tiles(bounds)

        assert data is not None, (
            "stitch_canopy_tiles returned None for the Spessart forest window.\n"
            "Possible causes:\n"
            "  1. Network unreachable (libdrive.ethz.ch DNS blocked).\n"
            "  2. GDAL vsicurl_streaming could not read the tile – check that "
            "GDAL_HTTP_TIMEOUT is sufficient and the server is reachable.\n"
            "  3. The ETH libdrive share token or URL has changed; verify "
            "_ETH_CANOPY_URL in download_raster.py."
        )
        assert prof is not None
        assert data.dtype == np.float32
        assert data.ndim == 2
        assert data.shape[0] > 0 and data.shape[1] > 0
        assert np.any(np.isfinite(data) & (data > 0)), (
            "Expected positive canopy heights over Spessart forest, "
            f"but got: unique vals = {np.unique(data[np.isfinite(data)])[:10]}"
        )
        for key in ("height", "width", "transform", "count", "dtype"):
            assert key in prof, f"Profile missing key '{key}'"


# ===========================================================================
# 7. Integration test – real WorldCover tile download (requires network)
# ===========================================================================

class TestRoughnessMapRealCoordinates:
    """Integration tests that download real WorldCover data from ESA S3.

    These tests are skipped by default.  Run with ``--integration`` to execute.
    """

    @pytest.mark.parametrize("lat,lon,label", [
        (52.52, 13.40, "Berlin, Germany"),
    ])
    def test_landcover_codes_and_z0_for_real_location(
        self, lat, lon, label, integration, tmp_path, capsys
    ):
        """Download WorldCover + ETH canopy tiles for *lat*/*lon*, compute z₀
        via the ORA bivariate model, and save a three-panel diagnostic figure
        (landcover + canopy height + roughness map).

        When ETH canopy tiles are unavailable for the area the ORA model falls
        back to the GWA4 table value for tree pixels and the figure shows two
        panels instead of three.

        Pass ``--integration`` to enable this test.
        """
        from terrain_fetcher.download_raster import (
            _WORLDCOVER_GRID_URL,
            _calculate_bounds,
            stitch_tiles,
            stitch_canopy_tiles,
            _align_to_reference,
            _compute_ora_z0_d,
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

        # --- identify and download WorldCover tiles ---
        # NOTE: gpd.read_file(grid_url) downloads a ~4 MB GeoJSON grid index
        # from ESA S3 – this single call accounts for most of the test latency.
        grid = gpd.read_file(grid_url)
        aoi = Polygon([(lon_, lat_) for lat_, lon_ in corners])
        tiles = grid[grid.intersects(aoi)].ll_tile.tolist()
        print(f"\n  WorldCover tiles required: {tiles}")

        data_lc, profile_lc = stitch_tiles(tiles, version, year, bounds)

        # --- download ETH canopy height tiles ---
        print("  Downloading ETH canopy height tiles...")
        data_canopy, profile_canopy = stitch_canopy_tiles(bounds)
        if data_canopy is not None:
            print(f"  Canopy tile shape: {data_canopy.shape}")
            h = _align_to_reference(data_canopy, profile_canopy, profile_lc)
        else:
            print("  No ETH canopy tiles available – ORA will use table fallback for trees.")
            h = None

        # --- compute z0 via ORA model ---
        lct = wk.get_landcover_table(table_name)
        lc_code_to_z0 = _build_lc_code_to_z0(table_name)
        z0_data, _ = _compute_ora_z0_d(data_lc, h, lct)

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
            data_lc, z0_data,
            title_prefix=f"{label}  ({side_km} km × {side_km} km)",
            out_stem=out_stem,
            canopy_data=h,
            want_canopy_panel=True,
        )
        out_path = _PLOTS_DIR / f"{out_stem}.png"
        assert out_path.exists(), f"Expected plot not found at {out_path}"

        captured = capsys.readouterr()
        assert label in captured.out


# ===========================================================================
# 8. Integration test – coordinates from the user's config file
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

    **Roughness strategy** (mirrors :func:`download_square_data`):

    * ``land_cover_table: GWA4`` (default) → ORA bivariate model; ETH canopy
      height tiles are also downloaded; diagnostic figure has **3 panels**
      (landcover + canopy height + roughness map).
    * Any other table (e.g. ``custom``) → straight vectorised lookup; figure
      has **2 panels** (landcover + roughness map).

    **Network latency** – the first step of any ``--integration`` run downloads
    the ESA WorldCover grid GeoJSON index (~4 MB from ESA S3) via
    ``geopandas.read_file(grid_url)``.  This is a one-time cost per test
    session and typically takes 5–30 s depending on your connection speed.
    Subsequent per-coordinate steps download the WorldCover tile(s) and, for
    GWA4, the matching ETH canopy tile(s).  The total delay is therefore
    **network-speed dependent** and not a code issue.

    Run::

        pytest tests/test_roughness_map.py --integration --config config.yaml -v -s
    """

    def test_landcover_and_roughness_from_config(self, request, capsys):
        """For each coordinate found in ``--config``, download WorldCover tiles,
        compute z₀ (ORA for GWA4; lookup for custom), and save diagnostic plots.

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
            stitch_canopy_tiles,
            _align_to_reference,
            _compute_ora_z0_d,
        )
        import geopandas as gpd
        import re
        import yaml
        from pathlib import Path as _Path
        from shapely.geometry import Polygon
        from terrain_fetcher.config import _DEFAULT_CUSTOM_TABLE_FILENAME

        # Read WorldCover settings from the config (fall back to defaults)
        with _Path(config_path).open() as fh:
            cfg_data = yaml.safe_load(fh) or {}
        table_name = cfg_data.get("land_cover_table", "GWA4")
        version = cfg_data.get("worldcover_version", "v100")
        year = int(cfg_data.get("worldcover_year", 2020))
        # Cap side_km at 20 km so integration tests stay fast regardless of
        # the value set in the config (which may be much larger, e.g. 50 km).
        side_km = min(float(cfg_data.get("side_km", 10.0)), 20.0)

        # Resolve the custom table path when table_name == "custom"
        custom_path: str | None = None
        if table_name == "custom":
            raw_path = cfg_data.get("custom_land_cover_table_path")
            if raw_path is not None:
                custom_path = str(_Path(raw_path))
            else:
                # Default: landcover_roughness.csv next to the config file
                custom_path = str(_Path(config_path).parent / _DEFAULT_CUSTOM_TABLE_FILENAME)

        # Build the lookup dict (used for breakdown tables and custom-table z0)
        lc_code_to_z0 = _build_lc_code_to_z0(table_name, custom_path=custom_path)

        # Load the full land-cover table object (needed for ORA model)
        if table_name == "custom":
            lct = load_custom_landcover_table(custom_path)
        else:
            lct = wk.get_landcover_table(table_name)

        use_ora = (table_name == "GWA4")

        # Download the grid GeoJSON once (main source of per-run latency).
        # NOTE: This is a ~4 MB file from ESA S3; see class docstring for details.
        grid_url = _WORLDCOVER_GRID_URL.format(version=version, year=year)
        print(f"\n  Fetching WorldCover grid index from: {grid_url}")
        grid = gpd.read_file(grid_url)

        failures: list[str] = []

        for lat, lon, label in coords:
            bounds, corners = _calculate_bounds(side_km, lat, lon)

            if table_name == "custom":
                lc_table_desc = f"custom file: {custom_path}"
            else:
                lc_table_desc = f"windkit.get_landcover_table('{table_name}')"
            _print_breakdown_header(
                f"Location  : {label}",
                extra_lines=[
                    f"Bounds    : {bounds}",
                    f"Area      : {side_km} km × {side_km} km",
                    f"Source    : ESA WorldCover {year} {version}",
                    f"Grid URL  : {grid_url}",
                    f"LC table  : {lc_table_desc}",
                    f"ORA model : {'yes (GWA4)' if use_ora else 'no (plain lookup)'}",
                ] + ([] if table_name == "custom" else [f"windkit   : {wk.__version__}"]),
            )

            aoi = Polygon([(lon_, lat_) for lat_, lon_ in corners])
            tiles = grid[grid.intersects(aoi)].ll_tile.tolist()
            print(f"\n  WorldCover tiles required: {tiles}")

            data_lc, profile_lc = stitch_tiles(tiles, version, year, bounds)

            # ---- z0 computation ----
            if use_ora:
                # GWA4: bivariate ORA model with ETH canopy height data
                print("  Downloading ETH canopy height tiles...")
                data_canopy, profile_canopy = stitch_canopy_tiles(bounds)
                if data_canopy is not None:
                    print(f"  Canopy tile shape: {data_canopy.shape}")
                    h = _align_to_reference(data_canopy, profile_canopy, profile_lc)
                else:
                    print(
                        "  No ETH canopy tiles available – "
                        "ORA will use table fallback for trees."
                    )
                    h = None
                z0_data, _ = _compute_ora_z0_d(data_lc, h, lct)
                canopy_vis = h
            else:
                # Non-GWA4 (custom): straight vectorised lookup
                z0_data = np.vectorize(
                    lambda c: float(lc_code_to_z0.get(int(c), np.nan))
                )(data_lc).astype(np.float32)
                canopy_vis = None

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

            # Save diagnostic plots (3-panel for GWA4/ORA, 2-panel for custom)
            out_stem = "config_" + re.sub(
                r"[^a-z0-9]+", "_", label.lower()
            ).strip("_")
            _plot_landcover_and_roughness(
                data_lc, z0_data,
                title_prefix=label,
                out_stem=out_stem,
                canopy_data=canopy_vis,
                want_canopy_panel=use_ora,
            )

        assert failures == [], "\n".join(failures)

        captured = capsys.readouterr()
        # At least the first coordinate's label must appear in the captured output
        assert coords[0][2] in captured.out
