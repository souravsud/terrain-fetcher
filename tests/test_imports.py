def test_fetchdata_imports():
    from fetchData import DownloadConfig, create_output_dir, download_raster_data
    from fetchData.csv_utils import load_coordinates_from_csv
    from fetchData.parameter_generation import generate_directions

    assert DownloadConfig is not None
    assert create_output_dir is not None
    assert download_raster_data is not None
    assert load_coordinates_from_csv is not None
    assert generate_directions is not None
