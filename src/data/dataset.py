"""PyTorch Dataset module for DocForge-VLM.

Handles loading of images, applying augmentations, and formatting data
for Qwen2-VL-2B-Instruct.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Try importing qwen_vl_utils, fallback if not available
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False

logger = logging.getLogger(__name__)

class DocForgeDataset(Dataset):
    """Dataset for training a Vision-Language Model on document forgery detection.
    
    Loads conversation JSON files and prepares images with optional augmentations.
    """
    
    def __init__(
        self,
        json_path: Union[str, Path],
        processor: Any = None,
        is_training: bool = False
    ):
        """Initializes the dataset.
        
        Args:
            json_path: Path to the JSON file containing conversation data.
            processor: The HuggingFace processor for the VLM (e.g., Qwen2VLProcessor).
                       Optional if you plan to use `collate_fn` differently.
            is_training: Whether to apply data augmentations.
        """
        self.json_path = Path(json_path)
        self.processor = processor
        self.is_training = is_training
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        logger.info(f"Loaded {len(self.data)} samples from {self.json_path}")
        
        # Define augmentations for training
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Gets a single dataset item.
        
        Args:
            idx: Index of the item.
            
        Returns:
            A dictionary containing the raw PIL image, conversation data, and label.
        """
        item = self.data[idx]
        image_path = Path(item["image_path"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a blank white image as fallback
            image = Image.new('RGB', (224, 224), color='white')
            
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "conversations": item["conversations"],
            "label": item["label"]
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]], processor: Any = None) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.
        
        Args:
            batch: List of items returned by __getitem__.
            processor: Qwen2VL processor. If None, it must be provided or handled externally.
            
        Returns:
            Dictionary of batched tensors ready for the model.
        """
        images = [item["image"] for item in batch]
        conversations = [item["conversations"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        
        # Format messages for Qwen2-VL
        messages = []
        for conv in conversations:
            formatted_conv = []
            for turn in conv:
                if turn["role"] == "user":
                    # Convert <image> placeholder to Qwen specific format
                    content = [{"type": "image"}, {"type": "text", "text": turn["content"].replace("<image>\\n", "")}]
                    formatted_conv.append({"role": "user", "content": content})
                else:
                    formatted_conv.append({"role": "assistant", "content": [{"type": "text", "text": turn["content"]}]})
            messages.append(formatted_conv)

        if processor is not None:
            # Apply Qwen chat template
            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                for msg in messages
            ]
            
            # Use qwen_vl_utils if available to process vision info
            if HAS_QWEN_UTILS:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = images
            
            batch_dict = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
            batch_dict["labels"] = labels
            return batch_dict
            
        # If no processor is passed, return raw structured data
        return {
            "images": images,
            "texts": messages,
            "labels": labels
        }

if __name__ == "__main__":
    # Simple test of the dataset if run directly
    logging.basicConfig(level=logging.INFO)
    dummy_json = Path("data/processed/train.json")
    if dummy_json.exists():
        dataset = DocForgeDataset(json_path=dummy_json, is_training=True)
        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
    else:
        print(f"No dummy JSON found at {dummy_json}. Run prepare.py first to test dataset.py.")
