# %% [markdown]
# # 🔍 DocForge-VLM: Document Forgery Detection with Fine-tuned Vision-Language Models
# 
# Fine-tuning **Qwen2-VL-2B-Instruct** with **QLoRA** (4-bit) for identity document tampering detection.
# 
# **Dataset**: SIDTD (Synthetic Identity Document Tampering Detection)  
# **Method**: LoRA (r=16, α=32) on attention projections  
# **Platform**: Kaggle T4 GPU  
# 
# ---

# %% [markdown]
# ## 1. Setup & Installation

# %%
import subprocess
import sys

packages = [
    "transformers>=4.45.0",
    "peft>=0.13.0",
    "bitsandbytes>=0.43.0",
    "trl>=0.12.0",
    "accelerate>=0.34.0",
    "qwen-vl-utils>=0.0.2",
    "scikit-learn",
    "seaborn",
    "huggingface_hub",
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Install SIDTD dataset package from GitHub
subprocess.check_call(["git", "clone", "--depth", "1",
    "https://github.com/Oriolrt/SIDTD_Dataset.git", "/tmp/SIDTD_Dataset"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "/tmp/SIDTD_Dataset", "-q"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# %%
import os
import gc
import json
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# HuggingFace login
from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if hf_token:
    login(token=hf_token)
    print("✅ Logged in to HuggingFace")
else:
    print("⚠️ No HF token found. Set HF_TOKEN as env var or Kaggle secret.")

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% [markdown]
# ## 2. Configuration

# %%
# ============================================================
# CONFIGURATION — Edit these values as needed
# ============================================================

# Model
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
BATCH_SIZE = 1               # Per-device batch size (keep small for T4)
GRAD_ACCUMULATION = 8        # Effective batch size = 1 * 8 = 8
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 1024

# Paths
OUTPUT_DIR = "./docforge_vlm_output"
RESULTS_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# %% [markdown]
# ## 3. Dataset Preparation
# 
# We use the **SIDTD** (Synthetic Identity Document Tampering Detection) dataset.
# It contains bonafide (authentic) and forged (tampered) identity document images.
# 
# The dataset will be downloaded or loaded from a Kaggle dataset input.

# %%
# ============================================================
# DATASET DOWNLOAD & LOADING
# ============================================================

# Method 1: Download using official SIDTD Python package
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

try:
    from SIDTD.data.DataLoader.Datasets import SIDTD as SIDTDDataset
    data = SIDTDDataset(download_original=False, custom_path_to_download=str(DATA_DIR)).download_dataset("templates")
    print("✅ SIDTD dataset downloaded via official package")
except Exception as e:
    print(f"⚠️ SIDTD package download failed: {e}")
    print("Falling back to Kaggle dataset path...")

# Method 2 (Fallback): Use dataset from Kaggle input
KAGGLE_DATASET_PATH = "/kaggle/input/sidtd-dataset"

def find_images(directory: Path) -> list[Path]:
    """Recursively find all image files in a directory."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(directory.rglob(ext))
    return sorted(set(images))


def load_sidtd_dataset(data_path: str) -> tuple[list[Path], list[int]]:
    """
    Load images and labels from SIDTD dataset.
    
    Handles multiple possible directory structures:
    1. data_path/bonafide/ + data_path/forged/
    2. data_path/SIDTD/bonafide/ + data_path/SIDTD/forged/
    3. data_path/authentic/ + data_path/tampered/
    4. Fallback: any images with 'forg' or 'fraud' in path are forged
    
    Returns:
        (image_paths, labels) where label 0=authentic, 1=tampered
    """
    data_path = Path(data_path)
    
    # Try common structures
    structures = [
        (data_path / "bonafide", data_path / "forged"),
        (data_path / "SIDTD" / "bonafide", data_path / "SIDTD" / "forged"),
        (data_path / "authentic", data_path / "tampered"),
        (data_path / "real", data_path / "fake"),
        (data_path / "Bonafide", data_path / "Forged"),
    ]
    
    for auth_dir, forg_dir in structures:
        if auth_dir.exists() and forg_dir.exists():
            auth_imgs = find_images(auth_dir)
            forg_imgs = find_images(forg_dir)
            if auth_imgs or forg_imgs:
                print(f"Found dataset structure: {auth_dir.name}/ + {forg_dir.name}/")
                print(f"  Authentic: {len(auth_imgs)} images")
                print(f"  Forged: {len(forg_imgs)} images")
                images = auth_imgs + forg_imgs
                labels = [0] * len(auth_imgs) + [1] * len(forg_imgs)
                return images, labels
    
    # Fallback: search recursively and classify by path
    print("Standard structure not found. Searching recursively...")
    all_imgs = find_images(data_path)
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {data_path}")
    
    images, labels = [], []
    for img in all_imgs:
        path_str = str(img).lower()
        if any(kw in path_str for kw in ["forg", "fraud", "fake", "tamper", "alter"]):
            labels.append(1)
        else:
            labels.append(0)
        images.append(img)
    
    n_auth = labels.count(0)
    n_forg = labels.count(1)
    print(f"  Found {len(images)} images ({n_auth} authentic, {n_forg} forged)")
    return images, labels


# Try loading
data_path = KAGGLE_DATASET_PATH
if not os.path.exists(data_path):
    # Try alternative paths (SIDTD package downloads to DATA_DIR)
    alternatives = [
        str(DATA_DIR),
        "/kaggle/input",
        "./data/raw",
        "./SIDTD",
    ]
    for alt in alternatives:
        if os.path.exists(alt):
            data_path = alt
            break
    else:
        print("⚠️ No dataset found. Please upload SIDTD as a Kaggle dataset.")
        print("   Or modify DATA_PATH to point to your dataset directory.")
        data_path = None

if data_path:
    images, labels = load_sidtd_dataset(data_path)
    print(f"\nTotal: {len(images)} images loaded")
else:
    images, labels = [], []
    print("Running in demo mode without data.")

# %% [markdown]
# ### 3.1 Create Train/Val/Test Splits

# %%
from sklearn.model_selection import train_test_split

if images:
    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    
    print(f"Train: {len(X_train)} ({sum(y_train)} forged, {len(y_train)-sum(y_train)} authentic)")
    print(f"Val:   {len(X_val)} ({sum(y_val)} forged, {len(y_val)-sum(y_val)} authentic)")
    print(f"Test:  {len(X_test)} ({sum(y_test)} forged, {len(y_test)-sum(y_test)} authentic)")
else:
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

# %% [markdown]
# ### 3.2 Format Data for VLM Training
# 
# Convert images + labels into conversation format for Qwen2-VL.

# %%
# ============================================================
# PROMPT TEMPLATES
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert in document forensics specializing in identity document "
    "verification. Analyze the provided document image for signs of forgery, "
    "tampering, or digital manipulation. Provide your verdict as AUTHENTIC or "
    "TAMPERED, followed by detailed reasoning."
)

USER_PROMPTS = [
    "Analyze this identity document for signs of forgery or tampering. Is it authentic or tampered?",
    "Examine this document image carefully. Determine if it is genuine or has been digitally manipulated.",
    "You are reviewing this identity document. Check for any signs of tampering, forgery, or digital manipulation.",
    "Inspect this ID card for authenticity. Is it a real document or a forgery?",
    "Conduct a forensic analysis of this document image. Can you identify any tampering?",
    "Assess this document for signs of digital alteration. State your verdict.",
    "Review the provided identity document and determine its authenticity.",
    "Look closely at the details of this document. Does it appear to be genuine or altered?",
]

AUTHENTIC_RESPONSES = [
    "VERDICT: AUTHENTIC\n\nAnalysis: After careful examination, this document appears to be genuine. The fonts are consistent throughout all text fields, the alignment of elements is proper, and there are no visible signs of digital tampering or splicing artifacts. The document's security features appear intact.",
    "VERDICT: AUTHENTIC\n\nAnalysis: This identity document passes forensic inspection. No inconsistencies detected in font styles, character spacing, or image compression patterns. The photograph region shows natural integration with the document background.",
    "VERDICT: AUTHENTIC\n\nAnalysis: The document exhibits consistent physical characteristics. Text fields show uniform typography, edge transitions are natural, and no splicing or copy-move artifacts are detected. I classify this as genuine.",
    "VERDICT: AUTHENTIC\n\nAnalysis: No evidence of manipulation found. The document's layout, text alignment, and color consistency are within expected parameters for a genuine identity document. No JPEG re-compression artifacts around text regions.",
    "VERDICT: AUTHENTIC\n\nAnalysis: This document appears unaltered. The text fields maintain consistent font weight, kerning, and baseline alignment. Background patterns and security elements show no interruption or digital editing traces.",
]

TAMPERED_RESPONSES = [
    "VERDICT: TAMPERED\n\nAnalysis: This document shows signs of digital manipulation. There are inconsistencies in the font style and weight within text fields that should be uniform. The character spacing appears irregular compared to standard document templates, suggesting text replacement.",
    "VERDICT: TAMPERED\n\nAnalysis: Evidence of forgery detected. The text region shows subtle compression artifacts inconsistent with the surrounding area, indicating that text fields have been digitally modified. Edge artifacts around altered characters are visible upon close inspection.",
    "VERDICT: TAMPERED\n\nAnalysis: This identity document has been altered. There is visible misalignment in certain text fields, and the font rendering differs from the expected template. These anomalies are consistent with digital text replacement tampering.",
    "VERDICT: TAMPERED\n\nAnalysis: Forensic analysis reveals manipulation. The document exhibits telltale signs of tampering including: inconsistent text baselines, slight color differences in modified fields, and unnatural edge transitions around the altered text regions.",
    "VERDICT: TAMPERED\n\nAnalysis: The document is not genuine. Anomalies detected include mismatched character rendering in personal data fields, irregular spacing patterns, and localized compression artifacts — all consistent with digital text field replacement.",
]


def format_sample_for_training(image_path: Path, label: int) -> dict:
    """Format a single sample into Qwen2-VL conversation format for SFTTrainer."""
    user_prompt = random.choice(USER_PROMPTS)
    response = random.choice(AUTHENTIC_RESPONSES) if label == 0 else random.choice(TAMPERED_RESPONSES)
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
        },
    ]
    
    return {"messages": messages, "label": label}


# Create formatted datasets
train_data = [format_sample_for_training(img, lbl) for img, lbl in zip(X_train, y_train)]
val_data = [format_sample_for_training(img, lbl) for img, lbl in zip(X_val, y_val)]
test_data = [format_sample_for_training(img, lbl) for img, lbl in zip(X_test, y_test)]

if train_data:
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    print(f"\n✅ Datasets created: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_data)} test")
else:
    train_dataset = val_dataset = None
    print("⚠️ No training data available.")

# %% [markdown]
# ## 4. Model Setup (QLoRA)
# 
# Load Qwen2-VL-2B-Instruct with 4-bit NF4 quantization and apply LoRA adapters.

# %%
print("=" * 60)
print("Loading model with 4-bit quantization...")
print("=" * 60)

# Quantization config for memory-efficient loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Prepare for QLoRA training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False

print("\n📊 Trainable Parameters:")
model.print_trainable_parameters()

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# %% [markdown]
# ## 5. Training
# 
# Fine-tune with SFTTrainer using gradient accumulation and mixed precision.

# %%
if train_dataset is not None and len(train_dataset) > 0:
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    # Training configuration
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field=None,       # We use formatting_func instead
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
        seed=SEED,
    )
    
    # Collate function for Qwen2-VL multimodal inputs
    def collate_fn(examples):
        texts = []
        image_inputs = []
        
        for example in examples:
            messages = example["messages"]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            
            # Extract images from messages
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            img = Image.open(item["image"]).convert("RGB")
                            # Resize if too large to save memory
                            max_dim = 1280
                            if max(img.size) > max_dim:
                                ratio = max_dim / max(img.size)
                                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                                img = img.resize(new_size, Image.LANCZOS)
                            image_inputs.append(img)
        
        # Process with the Qwen2-VL processor
        batch = processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        
        # Create labels (same as input_ids for causal LM, with padding masked)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train!
    train_result = trainer.train()
    
    # Save final model
    final_adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    processor.save_pretrained(final_adapter_path)
    
    # Save training metrics
    metrics = train_result.metrics
    with open(os.path.join(RESULTS_DIR, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Model saved to: {final_adapter_path}")
    
    # Plot training loss
    if hasattr(trainer.state, "log_history") and trainer.state.log_history:
        train_losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        if train_losses:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, color="#00D1B2", linewidth=2)
            plt.title("Training Loss", fontsize=14, fontweight="bold")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"), dpi=300, bbox_inches="tight")
            plt.show()

else:
    print("⚠️ Skipping training — no dataset available.")
    final_adapter_path = None

# %% [markdown]
# ## 6. Evaluation
# 
# Evaluate the fine-tuned model on the held-out test set.

# %%
def parse_verdict(response: str) -> str:
    """Parse the model's response to extract the verdict."""
    response_upper = response.upper()
    if "VERDICT: AUTHENTIC" in response_upper or "VERDICT:AUTHENTIC" in response_upper:
        return "AUTHENTIC"
    elif "VERDICT: TAMPERED" in response_upper or "VERDICT:TAMPERED" in response_upper:
        return "TAMPERED"
    elif "VERDICT: FORGED" in response_upper or "VERDICT:FORGED" in response_upper:
        return "TAMPERED"
    # Fallback: check if the words appear anywhere
    elif "AUTHENTIC" in response_upper and "TAMPERED" not in response_upper:
        return "AUTHENTIC"
    elif "TAMPERED" in response_upper or "FORGED" in response_upper:
        return "TAMPERED"
    return "UNKNOWN"


def run_inference(model, processor, image_path: str, prompt: str, device="cuda") -> str:
    """Run inference on a single image."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    img = Image.open(image_path).convert("RGB")
    max_dim = 1280
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )
    
    # Trim the prompt tokens from the output
    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return response


def evaluate_on_test_set(model, processor, test_data: list, device="cuda") -> dict:
    """Run evaluation on the full test set."""
    model.eval()
    y_true, y_pred = [], []
    responses = []
    
    eval_prompt = "Analyze this identity document for signs of forgery or tampering. Is it authentic or tampered?"
    
    print(f"Evaluating on {len(test_data)} test samples...")
    for i, sample in enumerate(test_data):
        try:
            true_label = sample["label"]
            y_true.append(true_label)
            
            # Get image path from the messages
            img_path = None
            for msg in sample["messages"]:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            img_path = item["image"]
                            break
            
            if img_path is None:
                y_pred.append(0)
                responses.append("ERROR: No image found")
                continue
            
            response = run_inference(model, processor, img_path, eval_prompt, device)
            responses.append(response)
            
            verdict = parse_verdict(response)
            pred_label = 0 if verdict == "AUTHENTIC" else 1
            y_pred.append(pred_label)
            
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(test_data)}] processed...")
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            y_pred.append(0)
            responses.append(f"ERROR: {e}")
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=["AUTHENTIC", "TAMPERED"], zero_division=0)
    
    results = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_samples": len(y_true),
        "n_correct": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)),
    }
    
    return results, y_true, y_pred, responses


# Run evaluation
if test_data:
    torch.cuda.empty_cache()
    gc.collect()
    
    results, y_true, y_pred, responses = evaluate_on_test_set(model, processor, test_data)
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision_macro']:.4f}")
    print(f"Recall:    {results['recall_macro']:.4f}")
    print(f"F1 Score:  {results['f1_macro']:.4f}")
    print(f"\n{results['classification_report']}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {RESULTS_DIR}/evaluation_results.json")
else:
    print("⚠️ Skipping evaluation — no test data available.")

# %% [markdown]
# ## 7. Visualization

# %%
if test_data and "y_true" in dir() and y_true:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")
    
    # --- Confusion Matrix ---
    ax1 = axes[0]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=["AUTHENTIC", "TAMPERED"],
        yticklabels=["AUTHENTIC", "TAMPERED"],
        ax=ax1, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold", color="white")
    ax1.set_ylabel("True Label", fontsize=12, color="white")
    ax1.set_xlabel("Predicted Label", fontsize=12, color="white")
    ax1.tick_params(colors="white")
    ax1.set_facecolor("#16213e")
    
    # --- Metrics Bar Chart ---
    ax2 = axes[1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_values = [
        results["accuracy"],
        results["precision_macro"],
        results["recall_macro"],
        results["f1_macro"],
    ]
    colors = ["#00D1B2", "#7B68EE", "#FF6B6B", "#FFD93D"]
    bars = ax2.bar(metric_names, metric_values, color=colors, edgecolor="white", linewidth=0.5)
    
    for bar, val in zip(bars, metric_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2%}", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="white",
        )
    
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Evaluation Metrics", fontsize=14, fontweight="bold", color="white")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("white")
    ax2.spines["left"].set_color("white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "evaluation_plots.png"), dpi=300, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    
    # --- Sample Predictions ---
    print("\n📝 Sample Predictions:")
    print("-" * 80)
    n_show = min(5, len(test_data))
    for i in range(n_show):
        true_lbl = "AUTHENTIC" if y_true[i] == 0 else "TAMPERED"
        pred_lbl = "AUTHENTIC" if y_pred[i] == 0 else "TAMPERED"
        correct = "✅" if y_true[i] == y_pred[i] else "❌"
        print(f"{correct} Sample {i+1}: True={true_lbl}, Predicted={pred_lbl}")
        print(f"   Response: {responses[i][:150]}...")
        print()

# %% [markdown]
# ## 8. Save & Export
# 
# Push the trained LoRA adapter to HuggingFace Hub (optional).

# %%
# ============================================================
# PUSH TO HUGGINGFACE HUB (Optional)
# ============================================================
# Uncomment the lines below and replace with your HF username
# 
# from huggingface_hub import login
# 
# # Login with your token (set HF_TOKEN as a Kaggle secret)
# hf_token = os.environ.get("HF_TOKEN")
# if hf_token:
#     login(token=hf_token)
#     
#     HF_REPO = "YOUR_USERNAME/DocForge-VLM-Qwen2-2B"
#     
#     model.push_to_hub(HF_REPO, use_auth_token=True)
#     processor.push_to_hub(HF_REPO, use_auth_token=True)
#     print(f"✅ Model pushed to: https://huggingface.co/{HF_REPO}")
# else:
#     print("Set HF_TOKEN as a Kaggle secret to push to Hub.")

# %%
# Final cleanup
print("\n" + "=" * 60)
print("🎉 DocForge-VLM Pipeline Complete!")
print("=" * 60)

if final_adapter_path:
    print(f"\n📁 Outputs:")
    print(f"   Adapter: {final_adapter_path}")
    print(f"   Results: {RESULTS_DIR}")
    
    # List output files
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"   {fpath} ({size_mb:.1f} MB)")
