# Literature Review: Document Forgery Detection with Vision-Language Models

## 1. Vision-Language Models for Document Understanding

### DAVE: A VLM Vision Encoder for Document Understanding (2025)
- **Key Idea**: Purpose-built vision encoder with structural and spatial awareness for documents
- **Relevance**: Shows that generic VLM encoders underperform on document tasks; domain-specific adaptation improves performance significantly
- **Takeaway**: Fine-tuning the language model side (our approach with LoRA) while leveraging pre-trained vision encoders is a valid strategy

### DocVLM: Make Your VLM an Efficient Reader (2024)
- **Key Idea**: Integrates OCR tokens to compress visual information, reducing computational overhead for high-resolution document images
- **Relevance**: Demonstrates that VLMs can be adapted for document-specific tasks with efficiency gains
- **Takeaway**: Our approach of using Qwen2-VL's native dynamic resolution handling avoids the need for separate OCR integration

---

## 2. Document Forgery & Tampering Detection

### AIForge-Doc: Benchmarking AI-Generated Document Forgeries (2026)
- **Key Idea**: Benchmark focusing on modern threats — diffusion-based and AI-inpainted forgeries in financial/form documents
- **Relevance**: Directly relevant to our fraud detection task; shows that traditional forgery detection methods struggle with AI-generated forgeries
- **Takeaway**: VLM-based approaches that understand document semantics (not just pixel-level artifacts) may be more robust against sophisticated forgeries

### ForensicFormer: Hierarchical Multi-Scale Reasoning for Forgery Detection (2026)
- **Key Idea**: Introduces hierarchical multi-scale reasoning for robust, cross-domain forgery detection
- **Relevance**: Demonstrates the importance of multi-scale analysis for detecting both coarse and fine-grained manipulations
- **Takeaway**: Our VLM approach implicitly captures multi-scale features through the vision encoder's hierarchical representations

### FFDN: Frequency-Feature Decomposition Network (2024)
- **Key Idea**: Uses Discrete Wavelet Transform (DWT) to detect subtle compression artifacts left by digital forgeries
- **Relevance**: Frequency-domain analysis catches artifacts invisible in spatial domain
- **Takeaway**: Our VLM approach focuses on semantic understanding rather than frequency analysis; these could be complementary approaches

---

## 3. Efficient Fine-tuning of VLMs

### LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- **Key Idea**: Inject trainable low-rank matrices into frozen pre-trained weights, reducing trainable parameters by ~10,000x
- **Relevance**: Core technique we use for efficient fine-tuning on consumer GPUs
- **Our Config**: r=16, alpha=32, targeting attention projections (q, k, v, o)

### QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
- **Key Idea**: Combine 4-bit NormalFloat quantization with LoRA for memory-efficient fine-tuning
- **Relevance**: Enables fine-tuning 2B+ parameter VLMs on a single T4 GPU (16GB)
- **Our Config**: NF4 quantization, double quantization, bfloat16 compute dtype

---

## 4. Datasets

### SIDTD: Synthetic Identity Document Tampering Detection
- **Source**: Zenodo (https://zenodo.org/records/7897381)
- **Content**: 573 bonafide + 573 forged identity document images
- **Forgery Type**: Text field replacement on MIDV-2020 documents
- **Why We Use It**: Standard benchmark, publicly available, realistic forgery types

### MIDV-2020
- **Source**: MIDV project
- **Content**: Video clips and scans of identity documents from multiple countries
- **Role**: Base dataset that SIDTD builds upon; provides authentic document samples

---

## 5. Our Approach: Why VLMs for Forgery Detection?

Traditional approaches (CNN-based, frequency analysis) detect **pixel-level artifacts** but struggle with:
- High-quality forgeries that leave minimal artifacts
- Semantic inconsistencies (wrong date format, impossible field combinations)
- Cross-document generalization

**Our VLM approach** brings:
1. **Semantic understanding**: The model understands what document fields *should* look like
2. **Explainability**: The model provides natural language reasoning for its decisions
3. **Flexibility**: Can be prompted for different analysis tasks without retraining
4. **Transfer learning**: Pre-trained knowledge about documents, text, and visual patterns

This represents a paradigm shift from "artifact detection" to "document understanding" for forgery detection.
