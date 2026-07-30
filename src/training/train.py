import os
import argparse
import logging
import json
import torch
import random
import numpy as np
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

# Add parent directory to path to allow imports when running as script
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from src.models.model import load_model, apply_lora
except ImportError:
    pass

logger = logging.getLogger(__name__)

def setup_training_args(
    output_dir: str, 
    epochs: int, 
    batch_size: int, 
    lr: float,
    max_seq_length: int = 1024
) -> SFTConfig:
    """
    Set up training arguments optimized for Kaggle T4 GPU.
    
    Args:
        output_dir (str): Directory to save model checkpoints.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size per device.
        lr (float): Learning rate.
        max_seq_length (int): Maximum sequence length.
        
    Returns:
        SFTConfig: Configuration object for SFTTrainer.
    """
    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16 // batch_size if batch_size > 0 else 8,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=bf16_available,
        fp16=not bf16_available,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        save_steps=100,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to='none',
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

def train(model, processor, train_dataset, val_dataset, training_args, output_dir):
    """
    Initialize trainer and run training.
    
    Args:
        model: The model to train.
        processor: The processor corresponding to the model.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        training_args: Training arguments.
        output_dir (str): Directory to save outputs.
        
    Returns:
        SFTTrainer: The trainer instance.
    """
    logger.info("Initializing SFTTrainer")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        peft_config=model.peft_config['default'] if hasattr(model, 'peft_config') else None,
    )
    
    logger.info("Starting training")
    train_result = trainer.train()
    
    logger.info("Saving best model")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Train DocForge-VLM")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for model checkpoints")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA R value")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha value")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    logger.info(f"Training with args: {args}")
    
    # Example dataset loading placeholder
    train_dataset = None
    val_dataset = None
    logger.warning("No datasets loaded. Training will not execute properly.")
    
    model, processor = load_model(args.model_name, quantize=True)
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    
    training_args = setup_training_args(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        max_seq_length=args.max_seq_length
    )
    
    if train_dataset is not None:
        train(model, processor, train_dataset, val_dataset, training_args, args.output_dir)

if __name__ == '__main__':
    main()
