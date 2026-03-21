from .download_config import DownloadConfig


def download_raster_data(lat, lon, index, out_dir, config):
    """Download DEM/roughness data for one coordinate pair using a config object."""
    from .download_raster import DEMDownloader

    downloader = DEMDownloader(config)
    return downloader.download_single_location(lat, lon, index, out_dir)


def create_output_dir(lat, lon, index, root_folder):
    """Create a location output folder if it does not already exist."""
    from .download_raster import create_output_dir as _create_output_dir

    return _create_output_dir(lat, lon, index, root_folder)


def download_square_data(*args, **kwargs):
    """Download DEM/roughness data for a square around one point."""
    from .download_raster import download_square_data as _download_square_data

    return _download_square_data(*args, **kwargs)


def __getattr__(name):
    if name == "DEMDownloader":
        from .download_raster import DEMDownloader

        return DEMDownloader
    raise AttributeError(f"module 'fetchData' has no attribute {name!r}")


__all__ = [
    "DEMDownloader",
    "DownloadConfig",
    "create_output_dir",
    "download_raster_data",
    "download_square_data",
]
