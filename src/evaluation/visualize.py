import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from PIL import Image

# Use a professional dark theme for publication-quality plots
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1c1c1c", "figure.facecolor": "#121212"})

def plot_confusion_matrix(y_true, y_pred, output_path: str):
    """Plot confusion matrix using Seaborn heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['AUTHENTIC', 'TAMPERED'],
                yticklabels=['AUTHENTIC', 'TAMPERED'],
                cbar_kws={'label': 'Count'},
                annot_kws={"size": 14})
    
    plt.title('Confusion Matrix: Forgery Detection', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, output_path: str):
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#00d2ff', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#888888', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(training_log_path: str, output_path: str):
    """Plot training and evaluation metrics from Hugging Face Trainer logs."""
    try:
        with open(training_log_path, 'r') as f:
            logs = json.load(f)
            
        train_steps = []
        train_loss = []
        eval_steps = []
        eval_loss = []
        eval_acc = []
        
        for entry in logs:
            if 'loss' in entry and 'eval_loss' not in entry:
                train_steps.append(entry.get('step', len(train_steps)))
                train_loss.append(entry['loss'])
            elif 'eval_loss' in entry:
                eval_steps.append(entry.get('step', len(eval_steps)))
                eval_loss.append(entry['eval_loss'])
                if 'eval_accuracy' in entry:
                    eval_acc.append(entry['eval_accuracy'])
                    
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(train_steps, train_loss, 'b-', label='Train Loss', alpha=0.7)
        ax1.plot(eval_steps, eval_loss, 'r-', label='Eval Loss', marker='o')
        ax1.set_xlabel('Steps', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14, color='white')
        ax1.tick_params('y', colors='white')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.2)
        
        if eval_acc:
            ax2 = ax1.twinx()
            ax2.plot(eval_steps, eval_acc, 'g--', label='Eval Accuracy', marker='s')
            ax2.set_ylabel('Accuracy', fontsize=14, color='white')
            ax2.tick_params('y', colors='white')
            ax2.legend(loc='upper right', fontsize=12)
            
        plt.title('Training Progression', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting training curves: {e}")

def plot_robustness_results(benchmark_results: dict, output_path: str):
    """Plot bar chart of accuracy vs perturbation type and severity."""
    perturbation_types = list(benchmark_results.keys())
    
    # We'll create subplots for each perturbation type
    n_types = len(perturbation_types)
    fig, axes = plt.subplots(1, n_types, figsize=(4*n_types, 5), sharey=True)
    if n_types == 1:
        axes = [axes]
        
    for idx, p_type in enumerate(perturbation_types):
        ax = axes[idx]
        results = benchmark_results[p_type]
        severities = list(results.keys())
        accuracies = list(results.values())
        
        # Sort if they are numeric
        try:
            sorted_items = sorted(zip(severities, accuracies), key=lambda x: float(x[0]))
            severities = [str(x[0]) for x in sorted_items]
            accuracies = [x[1] for x in sorted_items]
        except ValueError:
            pass
            
        bars = ax.bar(severities, accuracies, color='#00d2ff', alpha=0.8)
        ax.set_title(f'{p_type.capitalize()}', fontsize=14)
        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylim(0, 1.05)
        
        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=14)
            
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    plt.suptitle('Model Robustness under Perturbations', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_examples(images: list, predictions: list, labels: list, output_path: str, n: int = 8):
    """Plot grid of example predictions with correct/incorrect highlighting."""
    n = min(n, len(images))
    if n == 0:
        return
        
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten()
    
    for idx in range(n):
        ax = axes[idx]
        try:
            img = Image.open(images[idx]).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            
            true_label = labels[idx]
            pred_label = predictions[idx]
            is_correct = (true_label.upper() == pred_label.upper())
            
            color = '#00ff00' if is_correct else '#ff0000'
            title = f"True: {true_label}\nPred: {pred_label}"
            
            # Add a colored border to indicate correctness
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
                spine.set_visible(True)
                
            ax.set_title(title, color=color, fontsize=12, fontweight='bold', pad=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\nimage", ha='center', va='center')
            ax.axis('off')
            
    # Hide empty subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
