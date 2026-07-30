<h1 align="center">🔍 DocForge-VLM</h1>
<h3 align="center">Document Forgery Detection using Fine-tuned Vision-Language Models</h3>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Qwen2--VL-2B-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Qwen2-VL">
  <img src="https://img.shields.io/badge/QLoRA-4bit-00D1B2?style=for-the-badge" alt="QLoRA">
  <img src="https://img.shields.io/badge/PEFT-LoRA-7B68EE?style=for-the-badge" alt="PEFT">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>Fine-tuning Qwen2-VL-2B-Instruct with QLoRA for identity document tampering detection on the SIDTD benchmark.</b>
</p>

<p align="center">
  <a href="#-key-results">Results</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 🎯 Problem Statement

Digital identity verification systems process **billions of documents annually**, making them prime targets for fraud. Traditional rule-based and CNN-based forgery detection methods struggle with:

- **High-quality digital forgeries** that leave minimal pixel-level artifacts
- **Semantic inconsistencies** (wrong date formats, impossible field combinations) that require document understanding
- **Cross-document generalization** across different ID types and countries
- **Explainability** — providing actionable reasoning for fraud decisions

**DocForge-VLM** addresses these challenges by leveraging **Vision-Language Models (VLMs)** that understand both the visual appearance and semantic content of identity documents.

## ✨ Key Features

- 🧠 **VLM-based Detection**: Uses Qwen2-VL-2B to jointly reason over visual and textual document features
- ⚡ **Efficient Fine-tuning**: QLoRA (4-bit NF4 quantization + LoRA r=16) — only **~0.2% parameters trained**
- 📊 **Explainable Predictions**: Model provides natural language reasoning for each verdict
- 🛡️ **Robustness Benchmarks**: Tested against noise, compression, rotation, and brightness perturbations
- 🔬 **Research-grade Evaluation**: Comprehensive metrics, ablation studies, and adversarial testing

## 📈 Key Results

> ⚠️ **Work in Progress** — Results will be updated as training completes.

| Model | Accuracy | F1 (Macro) | Precision | Recall | ROC-AUC |
|-------|----------|------------|-----------|--------|---------|
| Qwen2-VL-2B (Zero-shot) | TBD | TBD | TBD | TBD | TBD |
| **DocForge-VLM (QLoRA r=16)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

### Robustness Benchmark

| Perturbation | Accuracy Drop |
|---|---|
| JPEG Compression (Q=30) | TBD |
| Gaussian Noise (σ=25) | TBD |
| Rotation (±10°) | TBD |
| Brightness (±30%) | TBD |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DocForge-VLM Pipeline                │
├─────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  Document    │    │  Qwen2-VL-2B-Instruct        │  │
│  │  Image       │───▶│  ┌─────────────────────────┐ │  │
│  └──────────────┘    │  │  Vision Encoder (SigLIP)│ │  │
│                      │  │  (Frozen)               │ │  │
│  ┌──────────────┐    │  └──────────┬──────────────┘ │  │
│  │  Prompt      │    │             │                │  │
│  │ "Analyze     │───▶│  ┌──────────▼─────────────┐ │  │
│  │  this doc..."│    │  │  Language Model (Qwen2) │ │  │
│  └──────────────┘    │  │  + LoRA Adapters (r=16) │ │  │
│                      │  │  (Trained, 4-bit NF4)   │ │  │
│                      │  └──────────┬──────────────┘ │  │
│                      └─────────────┼────────────────┘  │
│                                    │                   │
│                      ┌─────────────▼────────────────┐  │
│                      │  Output:                     │  │
│                      │  VERDICT: TAMPERED           │  │
│                      │  Analysis: Font inconsistency│  │
│                      │  in name field, compression  │  │
│                      │  artifacts near photo area...│  │
│                      └──────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### Why VLMs for Forgery Detection?

| Approach | Pixel Artifacts | Semantic Understanding | Explainability | Cross-doc Transfer |
|----------|:-:|:-:|:-:|:-:|
| CNN-based | ✅ | ❌ | ❌ | ⚠️ |
| Frequency Analysis | ✅ | ❌ | ❌ | ⚠️ |
| **VLM (Ours)** | ✅ | ✅ | ✅ | ✅ |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- [HuggingFace](https://huggingface.co/) account (free)

### Installation

```bash
git clone https://github.com/CRGaming78/DocForge-VLM.git
cd DocForge-VLM
pip install -r requirements.txt
```

### Inference (Single Image)

```python
from src.models.model import load_trained_model
from src.models.prompts import USER_PROMPTS, parse_verdict

# Load fine-tuned model
model, processor = load_trained_model("path/to/adapter")

# Prepare input
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "path/to/document.jpg"},
        {"type": "text", "text": USER_PROMPTS[0]}
    ]}
]

# Generate prediction
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=["path/to/document.jpg"], return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(output[0], skip_special_tokens=True)

verdict, confidence = parse_verdict(response)
print(f"Verdict: {verdict} | Confidence: {confidence:.2f}")
```

## 🏋️ Training

### Dataset: SIDTD (Synthetic Identity Document Tampering Detection)

- **Source**: [GitHub](https://github.com/Oriolrt/SIDTD_Dataset) / [CVC Repository](https://tc11.cvc.uab.es/datasets/SIDTD_1) (based on MIDV-2020)
- **Size**: 573 bonafide + 573 forged document images
- **Forgery Type**: Text field replacement (name, date, ID number manipulation)
- **Split**: 70% train / 15% val / 15% test (stratified)

### Training on Kaggle (Recommended)

1. Upload `scripts/kaggle_train.py` as a Kaggle notebook
2. Enable GPU (T4 x2)
3. Add your HuggingFace token as a Kaggle secret (`HF_TOKEN`)
4. Run all cells

### Training Locally

```bash
# 1. Download and prepare data (uses official SIDTD Python package)
python -m src.data.download --output_dir data/raw --method package
python -m src.data.prepare --raw_dir data/raw --output_dir data/processed

# 2. Train with QLoRA
python -m src.training.train \
    --data_dir data/processed \
    --output_dir outputs \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --lora_r 16

# 3. Evaluate
python -m src.evaluation.evaluate \
    --model_path outputs/best_model \
    --test_data data/processed/test.json \
    --output_dir results
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | Qwen2-VL-2B-Instruct | Best balance of capability & efficiency for single-GPU training |
| Quantization | 4-bit NF4 (QLoRA) | Reduces VRAM from ~8GB to ~3GB for model weights |
| LoRA Rank | 16 | Good balance of expressiveness & parameter efficiency |
| LoRA Alpha | 32 | Standard 2x rank ratio |
| LoRA Targets | q, k, v, o projections | Attention layers capture document structure reasoning |
| Learning Rate | 2e-4 | Standard for LoRA fine-tuning |
| Scheduler | Cosine with 10% warmup | Smooth convergence |
| Effective Batch Size | 16 (2 × 8 accumulation) | Stable gradients despite small per-device batch |
| Precision | bfloat16 | Better numerical stability than fp16 |

## 📊 Evaluation

### Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score (macro & per-class)
- **Calibration**: ROC-AUC, Precision-Recall curve
- **Robustness**: Performance under noise, compression, geometric, and photometric perturbations
- **Explainability**: Manual review of model reasoning quality

### Run Evaluation

```bash
# Standard evaluation
python -m src.evaluation.evaluate --model_path outputs/best_model --test_data data/processed/test.json

# With robustness benchmarks
python -m src.evaluation.evaluate --model_path outputs/best_model --test_data data/processed/test.json --run_robustness
```

## 📁 Project Structure

```
DocForge-VLM/
├── configs/
│   └── config.yaml              # Centralized hyperparameters
├── src/
│   ├── data/
│   │   ├── download.py          # Dataset download & setup
│   │   ├── prepare.py           # Data preprocessing & splits
│   │   └── dataset.py           # PyTorch Dataset for VLM training
│   ├── models/
│   │   ├── model.py             # Model loading with QLoRA
│   │   └── prompts.py           # Prompt templates & parsing
│   ├── training/
│   │   └── train.py             # Training script (SFTTrainer)
│   └── evaluation/
│       ├── evaluate.py          # Evaluation harness
│       └── visualize.py         # Result visualization
├── scripts/
│   └── kaggle_train.py          # Self-contained Kaggle training script
├── docs/
│   ├── literature_review.md     # Research paper survey
│   └── experiment_log.md        # Experiment tracking
├── requirements.txt
└── README.md
```

## 🔬 Research Context

This project draws from recent advances in:

- **Vision-Language Models**: Qwen2-VL, PaliGemma 2, Florence-2 for multimodal understanding
- **Efficient Fine-tuning**: LoRA (Hu et al., 2021), QLoRA (Dettmers et al., 2023)
- **Document Forensics**: AIForge-Doc (2026), ForensicFormer (2026), FFDN (2024)
- **Document Understanding**: DAVE (2025), DocVLM (2024)

See [`docs/literature_review.md`](docs/literature_review.md) for the full survey.

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Framework | PyTorch 2.1+ |
| Models | HuggingFace Transformers |
| Fine-tuning | PEFT (LoRA/QLoRA) |
| Training | TRL (SFTTrainer) |
| Quantization | bitsandbytes (4-bit NF4) |
| VLM | Qwen2-VL-2B-Instruct |
| Evaluation | scikit-learn, matplotlib, seaborn |
| Experiment Tracking | Weights & Biases (optional) |

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen2-VL) for the Qwen2-VL model
- [SIDTD Dataset](https://zenodo.org/records/7897381) creators for the benchmark
- [HuggingFace](https://huggingface.co/) for the transformers and PEFT libraries
- [MIDV-2020](https://doi.org/10.18287/2412-6179-CO-756) for the base document images
