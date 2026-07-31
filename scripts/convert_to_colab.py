"""Convert the Kaggle notebook to a Google Colab-compatible version."""
import json

# Load existing notebook
with open("notebooks/docforge_vlm_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update title cell to say Colab
nb["cells"][0]["source"] = [
    "# \ud83d\udd0d DocForge-VLM: Document Forgery Detection with Fine-tuned Vision-Language Models\n",
    "\n",
    "Fine-tuning **Qwen2-VL-2B-Instruct** with **QLoRA** (4-bit NF4) for identity document tampering detection on the **SIDTD** benchmark.\n",
    "\n",
    "**Repository**: [GitHub](https://github.com/CRGaming78/DocForge-VLM)\n",
    "\n",
    "| Component | Details |\n",
    "|---|---|\n",
    "| Base Model | Qwen2-VL-2B-Instruct |\n",
    "| Method | QLoRA (4-bit NF4 + LoRA r=16) |\n",
    "| Dataset | SIDTD (Synthetic Identity Document Tampering Detection) |\n",
    "| Platform | Google Colab (T4 GPU) |\n",
    "\n",
    "> **\u26a0\ufe0f Runtime Setup**: Go to **Runtime \u2192 Change runtime type \u2192 T4 GPU** before running!",
]

# Update the HF login cell to use Colab userdata instead of Kaggle secrets
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "kaggle_secrets" in src:
            cell["source"] = [
                "import os, gc, json, random, glob\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import torch\n",
                "from PIL import Image\n",
                "from datasets import Dataset\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, precision_recall_fscore_support,\n",
                "    confusion_matrix, classification_report, roc_auc_score, roc_curve\n",
                ")\n",
                "from transformers import (\n",
                "    Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig\n",
                ")\n",
                "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel\n",
                "from trl import SFTConfig, SFTTrainer\n",
                "\n",
                "# HuggingFace login\n",
                "from huggingface_hub import login\n",
                'hf_token = os.environ.get("HF_TOKEN")\n',
                "if not hf_token:\n",
                "    try:\n",
                "        from google.colab import userdata\n",
                '        hf_token = userdata.get("HF_TOKEN")\n',
                "    except Exception:\n",
                "        pass\n",
                "if not hf_token:\n",
                "    # Manual input fallback\n",
                '    hf_token = input("Enter your HuggingFace token: ").strip()\n',
                "if hf_token:\n",
                "    login(token=hf_token)\n",
                '    print("\\u2705 Logged in to HuggingFace")\n',
                "\n",
                "# Seeds\n",
                "SEED = 42\n",
                "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)\n",
                "\n",
                'print(f"PyTorch: {torch.__version__}")\n',
                'print(f"CUDA: {torch.cuda.is_available()}")\n',
                "if torch.cuda.is_available():\n",
                '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n',
                '    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")',
            ]
            break

# Also update HF push cell to use Colab-compatible approach
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "push_to_hub" in src and "CRGaming78" in src:
            cell["source"] = [
                "# Push to HuggingFace Hub\n",
                "if hf_token:\n",
                '    HF_REPO = "CRGaming78/DocForge-VLM-Qwen2-2B"\n',
                "\n",
                '    print(f"Pushing model to {HF_REPO}...")\n',
                "    model.push_to_hub(HF_REPO)\n",
                "    processor.push_to_hub(HF_REPO)\n",
                '    print(f"Model pushed to: https://huggingface.co/{HF_REPO}")\n',
                "else:\n",
                '    print("No HF token. Add it via Colab Secrets (key icon in sidebar).")\n',
                "\n",
                'print("=" * 60)\n',
                'print("DocForge-VLM Pipeline Complete!")\n',
                'print("=" * 60)\n',
                'print(f"Adapter: {final_adapter_path}")\n',
                'print(f"Results: {RESULTS_DIR}")',
            ]
            break

# Add Colab metadata
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10.0"},
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
}

# Save as Colab version
with open("notebooks/docforge_vlm_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=True, indent=1)

print("Colab notebook created successfully!")
print(f"Cells: {len(nb['cells'])}")
