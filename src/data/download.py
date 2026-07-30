"""Script to download and extract the SIDTD dataset.

This script handles downloading datasets from Zenodo or alternative sources
and extracting them for preparation.
"""

import logging
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
SIDTD_URL = "https://zenodo.org/records/7897381/files/SIDTD.zip"
# Add a fallback URL if needed. For now, we will use a dummy MIDV URL.
MIDV_URL = "https://example.com/midv2020_subset.zip"

class DownloadProgressBar(urllib.request.tqdm if 'tqdm' in globals() else object):
    """A progress bar for urllib downloads.
    
    If tqdm is not available, falls back to a simple print.
    """
    def __init__(self):
        try:
            from tqdm import tqdm
            self.pbar = None
            self.tqdm_available = True
        except ImportError:
            self.tqdm_available = False
            self.last_percent = 0

    def __call__(self, block_num, block_size, total_size):
        if self.tqdm_available:
            from tqdm import tqdm
            if self.pbar is None:
                self.pbar = tqdm(total=total_size, unit="B", unit_scale=True)
            self.pbar.update(block_size)
        else:
            downloaded = block_num * block_size
            if total_size > 0:
                percent = int(downloaded * 100 / total_size)
                if percent > self.last_percent + 10:
                    logger.info(f"Downloaded: {percent}%")
                    self.last_percent = percent

def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extracts a zip or tar archive to a specified directory.
    
    Args:
        archive_path: Path to the archive file.
        extract_dir: Directory where the contents will be extracted.
    """
    logger.info(f"Extracting {archive_path} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)
                
            safe_extract(tar_ref, extract_dir)
    else:
        logger.error(f"Unsupported archive format: {archive_path.suffix}")
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    logger.info("Extraction complete.")

def download_file(url: str, dest_path: Path) -> None:
    """Downloads a file from a URL to a destination path.
    
    Args:
        url: The URL to download from.
        dest_path: The path to save the downloaded file.
    """
    logger.info(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, DownloadProgressBar())
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise

def download_sidtd(data_dir: Union[str, Path] = "data/raw") -> None:
    """Downloads and extracts the SIDTD dataset.
    
    Args:
        data_dir: The directory to store the raw dataset.
    """
    data_dir = Path(data_dir)
    archive_path = data_dir / "SIDTD.zip"
    extract_path = data_dir / "SIDTD"
    
    if extract_path.exists() and any(extract_path.iterdir()):
        logger.info(f"SIDTD dataset already exists at {extract_path}. Skipping download.")
        return

    if not archive_path.exists():
        logger.info("SIDTD archive not found locally. Starting download...")
        download_file(SIDTD_URL, archive_path)
    else:
        logger.info(f"SIDTD archive found locally at {archive_path}.")
        
    extract_archive(archive_path, data_dir)

def download_midv2020_subset(data_dir: Union[str, Path] = "data/raw") -> None:
    """Downloads a subset of MIDV-2020 as a fallback.
    
    Args:
        data_dir: The directory to store the raw dataset.
    """
    data_dir = Path(data_dir)
    archive_path = data_dir / "midv2020_subset.zip"
    extract_path = data_dir / "MIDV2020_Subset"
    
    if extract_path.exists() and any(extract_path.iterdir()):
        logger.info(f"MIDV-2020 subset already exists at {extract_path}. Skipping download.")
        return

    if not archive_path.exists():
        logger.info("MIDV-2020 archive not found locally. Starting download...")
        download_file(MIDV_URL, archive_path)
    else:
        logger.info(f"MIDV-2020 archive found locally at {archive_path}.")
        
    extract_archive(archive_path, data_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download DocForge-VLM datasets.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory to store datasets")
    parser.add_argument("--fallback", action="store_true", help="Download MIDV-2020 fallback instead of SIDTD")
    args = parser.parse_args()
    
    if args.fallback:
        download_midv2020_subset(args.data_dir)
    else:
        download_sidtd(args.data_dir)
