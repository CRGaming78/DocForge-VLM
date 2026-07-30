"""Data processing and loading module for DocForge-VLM."""

from .download import download_sidtd, download_midv2020_subset
from .prepare import prepare_dataset
from .dataset import DocForgeDataset

__all__ = [
    "download_sidtd",
    "download_midv2020_subset",
    "prepare_dataset",
    "DocForgeDataset",
]
