import pytest
from pathlib import Path
import sys


@pytest.fixture(autouse=False)
def repo_root_on_path():
    """Ensure the repository root is importable so ``main`` can be imported."""
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    yield
    # No cleanup — keeping root on path is harmless within the test session.


def test_terrain_fetcher_imports():
    from terrain_fetcher import DownloadConfig, create_output_dir, download_raster_data, load_config
    from terrain_fetcher.csv_utils import load_coordinates_from_csv

    assert DownloadConfig is not None
    assert create_output_dir is not None
    assert download_raster_data is not None
    assert load_coordinates_from_csv is not None
    assert load_config is not None


def test_download_config_new_fields():
    from terrain_fetcher import DownloadConfig

    cfg = DownloadConfig()
    assert cfg.worldcover_version == "v100"
    assert cfg.worldcover_year == 2020
    assert cfg.land_cover_table == "GWA4"


def test_load_config_defaults(tmp_path):
    """load_config with an empty YAML file returns all DownloadConfig defaults."""
    from terrain_fetcher.config import load_config

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("{}\n")

    cfg = load_config(empty_yaml)
    assert cfg.dem_name == "glo_30"
    assert cfg.side_length_km == 50.0
    assert cfg.dst_area_or_point == "Point"
    assert cfg.dst_ellipsoidal_height is False
    assert cfg.include_roughness_map is False
    assert cfg.worldcover_version == "v100"
    assert cfg.worldcover_year == 2020
    assert cfg.land_cover_table == "GWA4"
    assert cfg.custom_land_cover_table_path is None
    assert cfg.save_raw_files is True
    assert cfg.verbose is True
    assert cfg.show_plots is False


def test_load_config_overrides(tmp_path):
    """load_config correctly maps all supported YAML keys."""
    from terrain_fetcher.config import load_config

    yaml_content = """
dem_name: nasadem
ellipsoidal_height: true
area_or_point: Area
side_km: 25.0
roughness_map: true
worldcover_version: v200
worldcover_year: 2021
land_cover_table: GWA3
save_raw_files: false
verbose: false
show_plots: true
"""
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(cfg_file)
    assert cfg.dem_name == "nasadem"
    assert cfg.dst_ellipsoidal_height is True
    assert cfg.dst_area_or_point == "Area"
    assert cfg.side_length_km == 25.0
    assert cfg.include_roughness_map is True
    assert cfg.worldcover_version == "v200"
    assert cfg.worldcover_year == 2021
    assert cfg.land_cover_table == "GWA3"
    assert cfg.save_raw_files is False
    assert cfg.verbose is False
    assert cfg.show_plots is True


def test_load_config_missing_file(tmp_path):
    """load_config raises FileNotFoundError for a non-existent path."""
    from terrain_fetcher.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "this_file_does_not_exist.yaml")



# ---------------------------------------------------------------------------
# Custom land-cover table tests
# ---------------------------------------------------------------------------

_SAMPLE_CSV = """\
# WorldCover -> z0 [m]
# class_id,z0,description
10,0.6,Tree cover
20,0.08,Shrubland
80,0.0002,Permanent water bodies
999,0.05,Unknown
"""

_SAMPLE_WHITESPACE = """\
# Whitespace-separated format
10  0.6   Tree cover
20  0.08  Shrubland
80  0.0002  Permanent water bodies
999  0.05  Unknown
"""


def test_load_custom_landcover_table_csv(tmp_path):
    """load_custom_landcover_table correctly parses a CSV file."""
    from terrain_fetcher.lc_table import load_custom_landcover_table

    csv_file = tmp_path / "lc.csv"
    csv_file.write_text(_SAMPLE_CSV)

    table = load_custom_landcover_table(csv_file)
    assert table[10] == {"z0": 0.6, "description": "Tree cover"}
    assert table[20] == {"z0": 0.08, "description": "Shrubland"}
    assert table[80] == {"z0": 0.0002, "description": "Permanent water bodies"}
    assert table[999] == {"z0": 0.05, "description": "Unknown"}


def test_load_custom_landcover_table_whitespace(tmp_path):
    """load_custom_landcover_table accepts whitespace-separated files."""
    from terrain_fetcher.lc_table import load_custom_landcover_table

    ws_file = tmp_path / "lc.txt"
    ws_file.write_text(_SAMPLE_WHITESPACE)

    table = load_custom_landcover_table(ws_file)
    assert table[10]["z0"] == 0.6
    assert table[20]["z0"] == 0.08


def test_load_custom_landcover_table_missing_file(tmp_path):
    """load_custom_landcover_table raises FileNotFoundError for a missing file."""
    from terrain_fetcher.lc_table import load_custom_landcover_table

    with pytest.raises(FileNotFoundError):
        load_custom_landcover_table(tmp_path / "nonexistent.csv")


def test_load_custom_landcover_table_empty_file(tmp_path):
    """load_custom_landcover_table raises ValueError when file has no data rows."""
    from terrain_fetcher.lc_table import load_custom_landcover_table

    empty = tmp_path / "empty.csv"
    empty.write_text("# only comments\n\n")

    with pytest.raises(ValueError):
        load_custom_landcover_table(empty)


def test_load_config_custom_table_default_path(tmp_path):
    """When land_cover_table: custom, path defaults to landcover_roughness.csv next to config."""
    from terrain_fetcher.config import load_config

    yaml_content = "land_cover_table: custom\n"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(cfg_file)
    assert cfg.land_cover_table == "custom"
    assert cfg.custom_land_cover_table_path == str(tmp_path / "landcover_roughness.csv")


def test_load_config_custom_table_explicit_path(tmp_path):
    """When custom_land_cover_table_path is set, that path is used."""
    from terrain_fetcher.config import load_config

    explicit_path = "/some/other/path/my_table.csv"
    yaml_content = f"land_cover_table: custom\ncustom_land_cover_table_path: {explicit_path}\n"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(cfg_file)
    assert cfg.land_cover_table == "custom"
    assert cfg.custom_land_cover_table_path == explicit_path


def test_load_config_non_custom_table_path_is_none(tmp_path):
    """custom_land_cover_table_path is None when land_cover_table is not 'custom'."""
    from terrain_fetcher.config import load_config

    yaml_content = "land_cover_table: GWA4\n"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(cfg_file)
    assert cfg.land_cover_table == "GWA4"
    assert cfg.custom_land_cover_table_path is None


def test_default_landcover_roughness_csv_ships_with_repo():
    """The default landcover_roughness.csv is present in the repository root."""
    repo_root = Path(__file__).parent.parent
    csv_path = repo_root / "landcover_roughness.csv"
    assert csv_path.exists(), f"Expected {csv_path} to exist"

    from terrain_fetcher.lc_table import load_custom_landcover_table

    table = load_custom_landcover_table(csv_path)
    # Verify all 11 ESA WorldCover classes plus the fallback are present
    expected_ids = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 999}
    assert expected_ids.issubset(set(table.keys()))
    # Spot-check a few z0 values
    assert table[10]["z0"] == 0.6    # Tree cover
    assert table[80]["z0"] == 0.0002  # Permanent water bodies
    assert table[999]["z0"] == 0.05   # Fallback


# ---------------------------------------------------------------------------
# main.py entry-point tests
# ---------------------------------------------------------------------------

def test_main_missing_config_file(tmp_path, repo_root_on_path):
    """main() returns exit code 1 when the config file does not exist."""
    from main import main

    rc = main([str(tmp_path / "nonexistent.yaml")])
    assert rc == 1


def test_main_no_coordinates(tmp_path, repo_root_on_path):
    """main() returns exit code 1 when neither csv nor lat/lon is provided."""
    from main import main

    cfg_file = tmp_path / "no_coords.yaml"
    cfg_file.write_text("dem_name: glo_30\n")

    rc = main([str(cfg_file)])
    assert rc == 1


def test_main_default_config_path(monkeypatch, tmp_path, repo_root_on_path):
    """main() uses 'config.yaml' as the default config path."""
    from main import main

    # Change working directory to tmp_path; no config.yaml present → should return 1
    monkeypatch.chdir(tmp_path)
    rc = main([])
    assert rc == 1

