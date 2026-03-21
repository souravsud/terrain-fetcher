import pytest
from pathlib import Path


def test_fetchdata_imports():
    from fetchData import DownloadConfig, create_output_dir, download_raster_data, load_config
    from fetchData.csv_utils import load_coordinates_from_csv
    from fetchData.parameter_generation import generate_directions

    assert DownloadConfig is not None
    assert create_output_dir is not None
    assert download_raster_data is not None
    assert load_coordinates_from_csv is not None
    assert generate_directions is not None
    assert load_config is not None


def test_download_config_new_fields():
    from fetchData import DownloadConfig

    cfg = DownloadConfig()
    assert cfg.worldcover_version == "v100"
    assert cfg.worldcover_year == 2020
    assert cfg.land_cover_table == "GWA4"


def test_load_config_defaults(tmp_path):
    """load_config with an empty YAML file returns all DownloadConfig defaults."""
    from fetchData.config import load_config

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
    assert cfg.save_raw_files is True
    assert cfg.verbose is True
    assert cfg.show_plots is False


def test_load_config_overrides(tmp_path):
    """load_config correctly maps all supported YAML keys."""
    from fetchData.config import load_config

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
    from fetchData.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "this_file_does_not_exist.yaml")


def test_load_config_partial(tmp_path):
    """load_config handles a YAML file with only some keys set."""
    from fetchData.config import load_config

    yaml_content = "side_km: 100.0\nroughness_map: true\n"
    cfg_file = tmp_path / "partial.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(cfg_file)
    assert cfg.side_length_km == 100.0
    assert cfg.include_roughness_map is True
    # Unspecified fields retain defaults
    assert cfg.dem_name == "glo_30"
    assert cfg.verbose is True
