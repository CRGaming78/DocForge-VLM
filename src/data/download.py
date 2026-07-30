"""Script to download and extract the SIDTD dataset.

Supports two methods:
1. Official SIDTD Python package (recommended for Kaggle)
2. Direct download from CVC repository (tc11.cvc.uab.es)

Dataset: https://github.com/Oriolrt/SIDTD_Dataset
"""

import logging
import os
import shutil
import subprocess
import sys
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
SIDTD_GITHUB_URL = "https://github.com/Oriolrt/SIDTD_Dataset.git"
CVC_DATASET_URL = "https://tc11.cvc.uab.es/datasets/SIDTD_1"


def download_via_package(data_dir: Union[str, Path] = "data/raw",
                         kind: str = "templates") -> None:
    """Download SIDTD using the official Python package.
    
    This is the recommended method. It clones the SIDTD_Dataset repo,
    installs the package, and uses its API to download the data.
    
    Args:
        data_dir: Directory to store the downloaded dataset.
        kind: Type of data to download. One of 'templates', 'clips',
              'clips_cropped', or 'all'. Default is 'templates' which
              downloads document template images (fastest, recommended).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if _check_existing_data(data_dir):
        logger.info("SIDTD dataset already exists. Skipping download.")
        return
    
    # Try importing SIDTD package first
    try:
        from SIDTD.data.DataLoader.Datasets import SIDTD as SIDTDDataset
        logger.info("SIDTD package found. Downloading dataset...")
    except ImportError:
        logger.info("SIDTD package not found. Installing from GitHub...")
        _install_sidtd_package()
        from SIDTD.data.DataLoader.Datasets import SIDTD as SIDTDDataset
    
    # Download using the package API
    logger.info(f"Downloading SIDTD dataset (kind={kind})...")
    try:
        data = SIDTDDataset(
            download_original=False,
            custom_path_to_download=str(data_dir)
        ).download_dataset(kind)
        logger.info(f"SIDTD dataset downloaded successfully to {data_dir}")
    except Exception as e:
        logger.error(f"Package download failed: {e}")
        logger.info("Trying alternative download method...")
        download_via_direct(data_dir)


def _install_sidtd_package() -> None:
    """Clone and install the SIDTD package from GitHub."""
    clone_dir = Path("/tmp/SIDTD_Dataset")
    
    if not clone_dir.exists():
        logger.info(f"Cloning SIDTD repository to {clone_dir}...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", SIDTD_GITHUB_URL, str(clone_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    
    logger.info("Installing SIDTD package...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", str(clone_dir), "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    logger.info("SIDTD package installed successfully.")


def download_via_direct(data_dir: Union[str, Path] = "data/raw") -> None:
    """Download SIDTD dataset directly from CVC repository.
    
    Fallback method if the Python package approach fails.
    
    Args:
        data_dir: Directory to store the downloaded dataset.
    """
    import urllib.request
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if _check_existing_data(data_dir):
        logger.info("SIDTD dataset already exists. Skipping download.")
        return
    
    logger.info(f"Attempting direct download from CVC repository...")
    logger.info(f"If this fails, please manually download from: {CVC_DATASET_URL}")
    logger.info("Or use the package method: download_via_package()")
    
    # Try known direct download URLs
    urls_to_try = [
        "https://zenodo.org/records/7897381/files/SIDTD.zip",
    ]
    
    for url in urls_to_try:
        try:
            archive_path = data_dir / "SIDTD.zip"
            logger.info(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, str(archive_path))
            
            logger.info(f"Extracting to {data_dir}...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Clean up archive
            archive_path.unlink()
            logger.info("Download and extraction complete.")
            return
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue
    
    logger.error(
        "All download methods failed. Please download manually:\n"
        f"  1. Visit: {CVC_DATASET_URL}\n"
        f"  2. Download the dataset\n"
        f"  3. Extract to: {data_dir}\n"
        "Or use Kaggle: upload the dataset and set the path accordingly."
    )


def _check_existing_data(data_dir: Path) -> bool:
    """Check if dataset images already exist in the directory."""
    image_exts = {'.jpg', '.jpeg', '.png'}
    for f in data_dir.rglob("*"):
        if f.suffix.lower() in image_exts:
            return True
    return False


def find_dataset_images(data_dir: Union[str, Path]) -> tuple[list[Path], list[int]]:
    """Find and classify images in the dataset directory.
    
    Handles multiple directory structures:
    1. data_dir/bonafide/ + data_dir/forged/
    2. data_dir/SIDTD/bonafide/ + data_dir/SIDTD/forged/
    3. Subdirectories with 'real'/'fake', 'authentic'/'tampered' naming
    4. Fallback: classify by path keywords
    
    Args:
        data_dir: Root directory containing the dataset.
        
    Returns:
        Tuple of (image_paths, labels) where label 0=authentic, 1=forged.
    """
    data_dir = Path(data_dir)
    
    # Try common directory structures
    structures = [
        ("bonafide", "forged"),
        ("Bonafide", "Forged"),
        ("authentic", "tampered"),
        ("real", "fake"),
        ("genuine", "fraudulent"),
    ]
    
    for auth_name, forg_name in structures:
        # Search at multiple depths
        for subdir in [data_dir, data_dir / "SIDTD", data_dir / "sidtd"]:
            auth_dir = subdir / auth_name
            forg_dir = subdir / forg_name
            
            if auth_dir.exists() and forg_dir.exists():
                auth_imgs = _find_images_in_dir(auth_dir)
                forg_imgs = _find_images_in_dir(forg_dir)
                
                if auth_imgs or forg_imgs:
                    logger.info(f"Found: {auth_dir.name}/ ({len(auth_imgs)}) + "
                               f"{forg_dir.name}/ ({len(forg_imgs)})")
                    images = auth_imgs + forg_imgs
                    labels = [0] * len(auth_imgs) + [1] * len(forg_imgs)
                    return images, labels
    
    # Fallback: classify by path keywords
    logger.warning("Standard structure not found. Classifying by path keywords...")
    all_imgs = _find_images_in_dir(data_dir)
    
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {data_dir}")
    
    images, labels = [], []
    forged_keywords = {"forg", "fraud", "fake", "tamper", "alter", "manipul"}
    auth_keywords = {"bona", "real", "genuine", "authentic", "original"}
    
    for img in all_imgs:
        path_lower = str(img).lower()
        if any(kw in path_lower for kw in forged_keywords):
            labels.append(1)
            images.append(img)
        elif any(kw in path_lower for kw in auth_keywords):
            labels.append(0)
            images.append(img)
    
    if not images:
        logger.warning("Could not classify images. Treating all as authentic.")
        images = all_imgs
        labels = [0] * len(all_imgs)
    
    logger.info(f"Found {len(images)} images "
               f"({labels.count(0)} authentic, {labels.count(1)} forged)")
    return images, labels


def _find_images_in_dir(directory: Path) -> list[Path]:
    """Recursively find all image files in a directory."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(directory.rglob(ext))
    return sorted(set(images))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download SIDTD dataset.")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                       help="Directory to store the dataset")
    parser.add_argument("--kind", type=str, default="templates",
                       choices=["templates", "clips", "clips_cropped", "all"],
                       help="Type of data to download")
    parser.add_argument("--method", type=str, default="package",
                       choices=["package", "direct"],
                       help="Download method: 'package' (recommended) or 'direct'")
    args = parser.parse_args()
    
    if args.method == "package":
        download_via_package(args.output_dir, args.kind)
    else:
        download_via_direct(args.output_dir)
