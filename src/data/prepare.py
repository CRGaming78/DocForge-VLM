"""Data preparation script for DocForge-VLM.

Creates stratified train/val/test splits and generates conversation-format
training data for the Vision-Language Model.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42

PROMPT_TEMPLATES = [
    "<image>\nAnalyze this identity document and determine if it is authentic or forged.",
    "<image>\nPlease inspect the provided identity document for any signs of tampering.",
    "<image>\nIs this a genuine ID or a forgery? Provide your analysis.",
    "<image>\nExamine this ID card. Can you verify its authenticity?",
    "<image>\nLook closely at this document. Has it been fraudulently altered?",
]

AUTH_RESPONSES = [
    "VERDICT: AUTHENTIC\n\nAnalysis: The document shows no obvious signs of digital manipulation or text field tampering. Fonts are consistent, and security features appear intact.",
    "VERDICT: AUTHENTIC\n\nAnalysis: No tampering detected. The text alignments, background textures, and typography match standard bonafide identity documents.",
    "VERDICT: AUTHENTIC\n\nAnalysis: I have examined the document and found it to be genuine. There are no anomalies in the text fields or photograph area.",
]

FORGED_RESPONSES = [
    "VERDICT: FORGED\n\nAnalysis: Tampering detected. There are inconsistencies in the text fields, such as mismatched fonts, uneven alignment, or artifacts from digital alteration.",
    "VERDICT: FORGED\n\nAnalysis: This document appears to be fraudulent. Signs of manipulation are present around the key information fields, indicating text replacement.",
    "VERDICT: FORGED\n\nAnalysis: I classify this as a forgery. The image shows evidence of tampering, likely via digital modification of the personal details.",
]

def find_images(directory: Path) -> List[Path]:
    """Recursively finds all images in a directory."""
    images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))
    return list(set(images))

def create_conversation(image_path: Path, is_forged: bool, output_dir: Path) -> Dict:
    """Creates a conversation record for a single image.
    
    Args:
        image_path: Path to the image.
        is_forged: Boolean indicating if the document is forged.
        output_dir: The base output directory to calculate relative paths if needed,
                    though absolute or dataset-relative paths are often preferred.
                    Here we use the absolute path as a string.
                    
    Returns:
        Dict representing the conversation data.
    """
    prompt = random.choice(PROMPT_TEMPLATES)
    
    if is_forged:
        response = random.choice(FORGED_RESPONSES)
        label = 1
    else:
        response = random.choice(AUTH_RESPONSES)
        label = 0
        
    conversation = {
        "image_path": str(image_path.absolute()),
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ],
        "label": label
    }
    
    return conversation

def prepare_dataset(raw_dir: Union[str, Path] = 'data/raw', output_dir: Union[str, Path] = 'data/processed') -> None:
    """Prepares the dataset, creates splits, and saves conversation JSONs.
    
    Args:
        raw_dir: Path to the raw dataset directory.
        output_dir: Path to save the processed JSON files.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(RANDOM_SEED)
    
    # Assuming SIDTD has 'bonafide' and 'forged' subdirectories.
    bonafide_dir = raw_dir / "SIDTD" / "bonafide"
    forged_dir = raw_dir / "SIDTD" / "forged"
    
    if not bonafide_dir.exists() and not forged_dir.exists():
        # Fallback if structure is different
        logger.warning(f"Expected 'bonafide' and 'forged' dirs not found directly under {raw_dir / 'SIDTD'}. Searching recursively...")
        all_images = find_images(raw_dir)
        # Dummy logic for fallback
        bonafide_imgs = [p for p in all_images if 'forg' not in str(p).lower() and 'fraud' not in str(p).lower()]
        forged_imgs = [p for p in all_images if 'forg' in str(p).lower() or 'fraud' in str(p).lower()]
    else:
        bonafide_imgs = find_images(bonafide_dir)
        forged_imgs = find_images(forged_dir)
        
    logger.info(f"Found {len(bonafide_imgs)} bonafide images and {len(forged_imgs)} forged images.")
    
    if not bonafide_imgs and not forged_imgs:
        logger.error("No images found. Please check the raw data directory.")
        return
        
    # Create labels
    X = bonafide_imgs + forged_imgs
    y = [0] * len(bonafide_imgs) + [1] * len(forged_imgs)
    
    # Stratified split: Train (70%), Val (15%), Test (15%)
    if train_test_split:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED)
    else:
        logger.warning("scikit-learn not found. Falling back to non-stratified random split.")
        data = list(zip(X, y))
        random.shuffle(data)
        n = len(data)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        X_train, y_train = zip(*train_data) if train_data else ([], [])
        X_val, y_val = zip(*val_data) if val_data else ([], [])
        X_test, y_test = zip(*test_data) if test_data else ([], [])

    def create_split_data(x_split, y_split):
        return [create_conversation(img, label == 1, output_dir) for img, label in zip(x_split, y_split)]
        
    train_conv = create_split_data(X_train, y_train)
    val_conv = create_split_data(X_val, y_val)
    test_conv = create_split_data(X_test, y_test)
    
    # Save JSON files
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_conv, f, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_conv, f, indent=2)
    with open(output_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_conv, f, indent=2)
        
    # Print statistics
    def print_stats(name, y_split):
        total = len(y_split)
        if total == 0:
            logger.info(f"{name} Split: 0 images")
            return
        forged_count = sum(y_split)
        auth_count = total - forged_count
        logger.info(f"{name} Split: {total} images ({auth_count} authentic, {forged_count} forged)")
        
    print_stats("Train", y_train)
    print_stats("Validation", y_val)
    print_stats("Test", y_test)
    
    logger.info(f"Dataset preparation complete. Files saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset for VLM training.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory with raw datasets")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save JSON splits")
    args = parser.parse_args()
    
    prepare_dataset(args.raw_dir, args.output_dir)
