# Experiment Log: DocForge-VLM

## Experiment Tracking

| # | Date | Description | LoRA r | LR | Epochs | Train Loss | Val Loss | Accuracy | F1 | Notes |
|---|------|-------------|--------|-----|--------|------------|----------|----------|-----|-------|
| 1 | | Baseline: Zero-shot Qwen2-VL-2B | - | - | - | - | - | | | No fine-tuning |
| 2 | | QLoRA r=16, initial run | 16 | 2e-4 | 3 | | | | | First training run |
| 3 | | QLoRA r=8 ablation | 8 | 2e-4 | 3 | | | | | Rank ablation |
| 4 | | LR sweep | 16 | 1e-4 | 3 | | | | | Lower learning rate |
| 5 | | Best config + extended | 16 | 2e-4 | 5 | | | | | More epochs |

---

## Experiment 1: Zero-shot Baseline
**Date**:  
**Hypothesis**: Qwen2-VL-2B without fine-tuning will have poor forgery detection accuracy since it wasn't trained for this specific task.  
**Setup**: Run inference on test set with default prompts.  
**Results**:  
**Analysis**:  

---

## Experiment 2: QLoRA Fine-tuning (r=16)
**Date**:  
**Hypothesis**: Fine-tuning with LoRA r=16 on SIDTD will significantly improve forgery detection accuracy.  
**Setup**:
- Model: Qwen2-VL-2B-Instruct, 4-bit quantized
- LoRA: r=16, alpha=32, dropout=0.05
- Training: 3 epochs, lr=2e-4, cosine schedule, warmup=10%
- Data: SIDTD train split (~400 images)

**Results**:  
**Analysis**:  

---

## Experiment 3: LoRA Rank Ablation (r=8)
**Date**:  
**Hypothesis**: Lower rank may underfit; comparing r=8 vs r=16.  
**Setup**: Same as Exp 2 but with r=8, alpha=16.  
**Results**:  
**Analysis**:  

---

## Key Findings
<!-- Fill in after experiments -->

## Failed Approaches & Lessons Learned
<!-- Document what didn't work and why -->
