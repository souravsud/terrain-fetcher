"""Tests for roughness map preparation.

These tests inspect and validate:
  1. The ESA WorldCover land-use classifications loaded for a given area.
  2. The z0 lookup table (and its source) used to convert those
     classifications into aerodynamic roughness length values.
  3. The code→z0 mapping itself – including the edge-cases that can produce
     "fishy" (NaN) roughness values in the output raster.

Run all offline unit tests::

    pytest tests/test_roughness_map.py -v -s

Run the optional network integration test as well::

    pytest tests/test_roughness_map.py -v -s --integration

"""

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

        captured = capsys.readouterr()
        assert label in captured.out
