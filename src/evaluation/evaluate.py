import os
import re
import json
import argparse
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur

def parse_model_output(response: str) -> Dict[str, Any]:
    """Extract verdict and reasoning from model text output."""
    verdict = "UNKNOWN"
    confidence = 0.0
    reasoning = ""

    # Try to find VERDICT: AUTHENTIC or VERDICT: TAMPERED
    verdict_match = re.search(r'(?i)verdict:\s*(authentic|tampered)', response)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    else:
        # Fallback to checking if the words exist in the first few lines
        lower_resp = response.lower()
        if "authentic" in lower_resp[:200] and "tampered" not in lower_resp[:200]:
            verdict = "AUTHENTIC"
        elif "tampered" in lower_resp[:200] and "authentic" not in lower_resp[:200]:
            verdict = "TAMPERED"

    # Try to find Confidence: X%
    conf_match = re.search(r'(?i)confidence:\s*(\d+(?:\.\d+)?)%', response)
    if conf_match:
        confidence = float(conf_match.group(1)) / 100.0

    # Extract reasoning (everything after Analysis: or Reasoning:)
    reasoning_match = re.search(r'(?i)(?:analysis|reasoning):\s*(.*)', response, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = response.strip()

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning
    }

def compute_metrics(y_true: List[int], y_pred: List[int], y_scores: Optional[List[float]] = None) -> Dict[str, Any]:
    """Compute classification metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # ROC-AUC if scores are provided
    if y_scores is not None and len(set(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except Exception:
            metrics['roc_auc'] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=['AUTHENTIC', 'TAMPERED'], output_dict=True, zero_division=0
    )
    
    return metrics

def evaluate_model(model: Any, processor: Any, test_data: List[Dict[str, Any]], device: str = 'cuda') -> Dict[str, Any]:
    """Run inference on test set and compute metrics."""
    y_true = []
    y_pred = []
    y_scores = []
    results = []
    
    label_map = {"AUTHENTIC": 0, "TAMPERED": 1}

    model.eval()
    for item in tqdm(test_data, desc="Evaluating"):
        image_path = item['image_path']
        true_label = item['label'].upper()
        y_true.append(label_map.get(true_label, 0))
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Analyze this identity document and determine if it is authentic or tampered. Provide your verdict in the format 'VERDICT: AUTHENTIC' or 'VERDICT: TAMPERED', followed by 'Analysis: ' and your detailed reasoning."}
                ]}
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            parsed = parse_model_output(output_text)
            pred_label = parsed['verdict']
            
            pred_idx = label_map.get(pred_label, 0) # Default to 0 if unknown
            y_pred.append(pred_idx)
            
            # Use confidence as score if available, otherwise just use discrete prediction
            score = parsed['confidence'] if pred_idx == 1 else 1.0 - parsed['confidence']
            if score == 0.0:
                score = float(pred_idx)
            y_scores.append(score)
            
            results.append({
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": pred_label,
                "raw_output": output_text,
                "parsed": parsed
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Append defaults so arrays stay aligned
            y_pred.append(0)
            y_scores.append(0.0)
    
    metrics = compute_metrics(y_true, y_pred, y_scores)
    
    return {
        "metrics": metrics,
        "results": results
    }

def apply_perturbation(image: Image.Image, p_type: str, severity: Any) -> Image.Image:
    """Apply a specific perturbation to an image."""
    if p_type == 'jpeg':
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=severity)
        img_byte_arr.seek(0)
        return Image.open(img_byte_arr).convert("RGB")
    elif p_type == 'noise':
        img_t = TF.to_tensor(image)
        noise = torch.randn_like(img_t) * (severity / 255.0)
        img_t = torch.clamp(img_t + noise, 0, 1)
        return TF.to_pil_image(img_t)
    elif p_type == 'rotation':
        return TF.rotate(image, severity)
    elif p_type == 'brightness':
        # severity is percentage change, e.g., 30 for +30%, -30 for -30%
        factor = 1.0 + (severity / 100.0)
        return TF.adjust_brightness(image, factor)
    return image

def run_robustness_benchmark(model: Any, processor: Any, test_data: List[Dict[str, Any]], device: str = 'cuda') -> Dict[str, Any]:
    """Test model robustness under various perturbations."""
    perturbations = {
        'jpeg': [50, 30, 10], # Quality levels
        'noise': [10, 25, 50], # Sigma levels
        'rotation': [-10, -5, 5, 10], # Degrees
        'brightness': [-30, 30] # Percentage
    }
    
    benchmark_results = {}
    label_map = {"AUTHENTIC": 0, "TAMPERED": 1}
    
    model.eval()
    
    for p_type, severities in perturbations.items():
        benchmark_results[p_type] = {}
        for severity in severities:
            print(f"Running robustness test: {p_type} (severity: {severity})")
            y_true = []
            y_pred = []
            
            for item in tqdm(test_data, desc=f"{p_type}_{severity}"):
                image_path = item['image_path']
                true_label = item['label'].upper()
                y_true.append(label_map.get(true_label, 0))
                
                try:
                    image = Image.open(image_path).convert("RGB")
                    perturbed_image = apply_perturbation(image, p_type, severity)
                    
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image", "image": perturbed_image},
                            {"type": "text", "text": "Analyze this identity document and determine if it is authentic or tampered. Provide your verdict in the format 'VERDICT: AUTHENTIC' or 'VERDICT: TAMPERED'."}
                        ]}
                    ]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=128)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]

                    parsed = parse_model_output(output_text)
                    pred_idx = label_map.get(parsed['verdict'], 0)
                    y_pred.append(pred_idx)
                    
                except Exception as e:
                    y_pred.append(0)
                    
            acc = accuracy_score(y_true, y_pred)
            benchmark_results[p_type][str(severity)] = acc
            print(f"Accuracy for {p_type} ({severity}): {acc:.4f}")
            
    return benchmark_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate DocForge-VLM Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data JSON list")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--run_robustness", action="store_true", help="Run robustness benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dummy imports for main script functionality when run directly
    try:
        from transformers import AutoProcessor
        from peft import AutoPeftModelForCausalLM
        from qwen_vl_utils import process_vision_info
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {args.model_path} on {device}...")
        
        # Load model and processor
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print(f"Loading test data from {args.test_data}...")
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
            
        print("Starting evaluation...")
        eval_results = evaluate_model(model, processor, test_data, device=device)
        
        with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(eval_results, f, indent=4)
            
        print("Evaluation Complete. Metrics:")
        print(json.dumps(eval_results["metrics"], indent=4))
        
        if args.run_robustness:
            print("\nStarting Robustness Benchmark...")
            robustness_results = run_robustness_benchmark(model, processor, test_data, device=device)
            with open(os.path.join(args.output_dir, "robustness_results.json"), "w") as f:
                json.dump(robustness_results, f, indent=4)
            print("Robustness Benchmark Complete.")
            print(json.dumps(robustness_results, indent=4))
            
    except ImportError as e:
        print(f"Missing dependency: {e}. Please install required packages.")

if __name__ == "__main__":
    main()
