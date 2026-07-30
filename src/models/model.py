import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

logger = logging.getLogger(__name__)

def load_model(model_name: str = 'Qwen/Qwen2-VL-2B-Instruct', quantize: bool = True):
    """
    Load Qwen2-VL model and processor with optional 4-bit quantization.

    Args:
        model_name (str): Model name or path.
        quantize (bool): Whether to use 4-bit quantization.

    Returns:
        tuple: (model, processor)
    """
    logger.info(f"Loading processor from {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)

    quantization_config = None
    if quantize:
        logger.info("Initializing BitsAndBytesConfig for 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        )

    logger.info(f"Loading model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    return model, processor


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """
    Apply LoRA to the model.

    Args:
        model: The base model.
        r (int): LoRA rank.
        alpha (int): LoRA alpha.
        dropout (float): LoRA dropout.

    Returns:
        PeftModel: The model with LoRA applied.
    """
    logger.info(f"Configuring LoRA with r={r}, alpha={alpha}, dropout={dropout}")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    logger.info("Applying LoRA to the model")
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing to save memory on Kaggle T4
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
    model.print_trainable_parameters()
    return model


def load_trained_model(adapter_path: str, base_model: str = 'Qwen/Qwen2-VL-2B-Instruct'):
    """
    Load base model with a trained LoRA adapter for inference.

    Args:
        adapter_path (str): Path to the trained LoRA adapter.
        base_model (str): The base model name.

    Returns:
        tuple: (model, processor)
    """
    logger.info(f"Loading base model {base_model} and processor")
    processor = AutoProcessor.from_pretrained(base_model)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, processor
