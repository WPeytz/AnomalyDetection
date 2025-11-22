import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from pathlib import Path
import json
import os
import cv2
from PIL import Image


def compare_sam_mask_with_groundtruth(
    sam_mask: np.ndarray,
    groundtruth_mask_path: str,
    image: np.ndarray,
    save_path: str = None,
    title: str = "SAM vs Ground Truth Comparison"
):
    """
    Compare SAM predicted mask with ground truth PNG mask and visualize results.
    
    Args:
        sam_mask: Binary mask predicted by SAM (H, W) - boolean or 0/1 values
        groundtruth_mask_path: Path to ground truth mask PNG file
        image: Original RGB image (H, W, 3) - values in range [0, 255] or [0, 1]
        save_path: Path to save the comparison plot (if None, displays instead)
        title: Title for the plot
    
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Load ground truth mask
    gt_mask = Image.open(groundtruth_mask_path).convert('L')
    gt_mask = np.array(gt_mask)
    
    # Normalize ground truth mask to binary (0 or 1)
    gt_mask_binary = (gt_mask > 127).astype(np.uint8)
    
    # Ensure SAM mask is binary
    sam_mask_binary = (sam_mask > 0.5).astype(np.uint8) if sam_mask.dtype == float else sam_mask.astype(np.uint8)
    
    # Resize masks to match if needed
    if sam_mask_binary.shape != gt_mask_binary.shape:
        print(f"Resizing masks: SAM {sam_mask_binary.shape} -> GT {gt_mask_binary.shape}")
        sam_mask_binary = cv2.resize(sam_mask_binary, (gt_mask_binary.shape[1], gt_mask_binary.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
    
    # Ensure image is in correct format
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Resize image to match masks if needed
    if image.shape[:2] != gt_mask_binary.shape:
        image = cv2.resize(image, (gt_mask_binary.shape[1], gt_mask_binary.shape[0]))
    
    # Calculate metrics
    # Pixel-wise accuracy
    accuracy = np.mean(sam_mask_binary == gt_mask_binary)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((sam_mask_binary == 1) & (gt_mask_binary == 1))
    fp = np.sum((sam_mask_binary == 1) & (gt_mask_binary == 0))
    tn = np.sum((sam_mask_binary == 0) & (gt_mask_binary == 0))
    fn = np.sum((sam_mask_binary == 0) & (gt_mask_binary == 1))
    
    # Precision, Recall, F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU (Intersection over Union)
    intersection = np.sum((sam_mask_binary == 1) & (gt_mask_binary == 1))
    union = np.sum((sam_mask_binary == 1) | (gt_mask_binary == 1))
    iou = intersection / union if union > 0 else 0
    
    # Dice Coefficient
    dice = 2 * intersection / (np.sum(sam_mask_binary) + np.sum(gt_mask_binary)) if (np.sum(sam_mask_binary) + np.sum(gt_mask_binary)) > 0 else 0
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(gt_mask_binary, alpha=0.5, cmap='Reds')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # SAM predicted mask
    axes[0, 2].imshow(image)
    axes[0, 2].imshow(sam_mask_binary, alpha=0.5, cmap='Blues')
    axes[0, 2].set_title('SAM Predicted Mask', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Overlay comparison
    axes[1, 0].imshow(image)
    # Green for correct predictions, Red for GT only, Blue for SAM only
    overlay = np.zeros((*gt_mask_binary.shape, 3), dtype=np.uint8)
    overlay[tp_mask := ((sam_mask_binary == 1) & (gt_mask_binary == 1))] = [0, 255, 0]  # Green - True Positive
    overlay[fp_mask := ((sam_mask_binary == 1) & (gt_mask_binary == 0))] = [0, 0, 255]  # Blue - False Positive
    overlay[fn_mask := ((sam_mask_binary == 0) & (gt_mask_binary == 1))] = [255, 0, 0]  # Red - False Negative
    axes[1, 0].imshow(overlay, alpha=0.5)
    axes[1, 0].set_title('Overlay (Green=TP, Red=FN, Blue=FP)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference map
    diff_map = np.abs(sam_mask_binary.astype(float) - gt_mask_binary.astype(float))
    axes[1, 1].imshow(diff_map, cmap='hot')
    axes[1, 1].set_title('Difference Map', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Metrics text
    axes[1, 2].axis('off')
    metrics_text = f"""
    Pixel Accuracy: {accuracy:.4f}
    
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1-Score: {f1_score:.4f}
    
    IoU: {iou:.4f}
    Dice Coefficient: {dice:.4f}
    
    True Positives: {tp}
    False Positives: {fp}
    True Negatives: {tn}
    False Negatives: {fn}
    """
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].set_title('Metrics', fontsize=12, fontweight='bold')
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved SAM comparison plot to {save_path}")
        print(f"Accuracy: {accuracy:.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")
    else:
        plt.show()
    
    plt.close()
    
    # Return metrics dictionary
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'iou': float(iou),
        'dice': float(dice),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }


def plot_anomaly_scores(anomaly_scores_path: str, output_dir: str = None):
    """
    Generate comprehensive analysis plots for anomaly detection results.
    
    Args:
        anomaly_scores_path: Path to anomaly_scores.npz file (e.g., 'results/hazelnut_sam_fewshot_1/anomaly_scores.npz')
        output_dir: Directory to save plots (if None, saves to same directory as input file)
    """
    # Load data - works with both Few_shot_anomaly_detection and all_models_script outputs
    data = np.load(anomaly_scores_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(anomaly_scores_path).parent / "analysis_plots"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading anomaly scores from: {anomaly_scores_path}")
    print(f"Saving plots to: {output_dir}")
    print(f"\nAvailable keys: {data.files}")

    # Check if this is from few-shot script (has n_shots, seed) or all_models_script
    if "n_shots" in data.files:
        print(f"Shots: {data['n_shots']}")
        print(f"Seed: {data['seed']}")
    
    # Print k values
    if "k_cosine" in data.files:
        print(f"k_cosine: {data['k_cosine']}")
        print(f"k_euclidean: {data['k_euclidean']}")
        print(f"k_knn: {data['k_knn']}")

    # Load all three metrics
    scores_cosine = data["scores_cosine"]
    scores_euclidean = data["scores_euclidean"]
    scores_knn = data["scores_knn"]
    labels = data["labels"]

    print(f"\nCosine scores shape: {scores_cosine.shape}")
    print(f"Euclidean scores shape: {scores_euclidean.shape}")
    print(f"k-NN scores shape: {scores_knn.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Normal samples: {np.sum(labels == 0)}, Anomaly samples: {np.sum(labels == 1)}")

    metrics_dict = {
        'Cosine Similarity': scores_cosine,
        'Euclidean Distance': scores_euclidean,
        'k-NN Distance': scores_knn
    }

    # 1. Score Distribution (Histograms)
    print("\nGenerating score distribution plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (metric_name, scores) in enumerate(metrics_dict.items()):
        axes[idx].hist(scores[labels == 0], bins=20, alpha=0.7, label="Normal", color='blue')
        axes[idx].hist(scores[labels == 1], bins=20, alpha=0.7, label="Anomaly", color='red')
        axes[idx].legend()
        axes[idx].set_xlabel("Anomaly Score", fontsize=11)
        axes[idx].set_ylabel("Count", fontsize=11)
        axes[idx].set_title(f"Score Distribution - {metric_name}", fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "01_score_distributions.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / '01_score_distributions.png'}")
    plt.close()

    # 2. Density Plots
    print("Generating density plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (metric_name, scores) in enumerate(metrics_dict.items()):
        sns.kdeplot(scores[labels == 0], fill=True, label="Normal", ax=axes[idx], color='blue')
        sns.kdeplot(scores[labels == 1], fill=True, label="Anomaly", ax=axes[idx], color='red')
        axes[idx].set_xlabel("Anomaly Score", fontsize=11)
        axes[idx].set_ylabel("Density", fontsize=11)
        axes[idx].set_title(f"Score Density - {metric_name}", fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "02_score_densities.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / '02_score_densities.png'}")
    plt.close()

    # 3. Scatter Plots (sorted by label for clarity)
    print("Generating scatter plots...")
    # Sort samples: normal first, then anomalous
    sort_idx = np.argsort(labels)
    labels_sorted = labels[sort_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (metric_name, scores) in enumerate(metrics_dict.items()):
        scores_sorted = scores[sort_idx]
        scatter = axes[idx].scatter(range(len(scores_sorted)), scores_sorted, c=labels_sorted, cmap="coolwarm", alpha=0.7, s=30)
        axes[idx].set_xlabel("Sample Index (Normal → Anomaly)", fontsize=11)
        axes[idx].set_ylabel("Anomaly Score", fontsize=11)
        axes[idx].set_title(f"Scores per Sample - {metric_name}", fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[idx], label='Label (0=Normal, 1=Anomaly)')
    plt.tight_layout()
    plt.savefig(output_dir / "03_score_scatter.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / '03_score_scatter.png'}")
    plt.close()

    # 4. ROC Curves (all on one plot for comparison)
    print("Generating ROC curves...")
    plt.figure(figsize=(8, 6))
    for metric_name, scores in metrics_dict.items():
        fpr, tpr, _ = roc_curve(labels, scores) 
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{metric_name} (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0,1], [0,1], "--", color='gray', label='Random', linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - All Metrics", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "04_roc_curves.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / '04_roc_curves.png'}")
    plt.close()

    # 5. Precision-Recall Curves (all on one plot for comparison)
    print("Generating precision-recall curves...")
    plt.figure(figsize=(8, 6))
    for metric_name, scores in metrics_dict.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        plt.plot(recall, precision, label=metric_name, linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves - All Metrics", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "05_precision_recall_curves.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / '05_precision_recall_curves.png'}")
    plt.close()
    
    # 6. Box plots comparing metrics
    print("Generating box plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot for normal samples
    normal_data = [scores[labels == 0] for scores in metrics_dict.values()]
    axes[0].boxplot(normal_data, tick_labels=metrics_dict.keys())
    axes[0].set_ylabel("Anomaly Score", fontsize=11)
    axes[0].set_title("Normal Samples - Score Distribution", fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Box plot for anomaly samples
    anomaly_data = [scores[labels == 1] for scores in metrics_dict.values()]
    axes[1].boxplot(anomaly_data, tick_labels=metrics_dict.keys())
    axes[1].set_ylabel("Anomaly Score", fontsize=11)
    axes[1].set_title("Anomaly Samples - Score Distribution", fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_box_plots.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / '06_box_plots.png'}")
    plt.close()
    
    # 7. Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_file = output_dir / "summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Anomaly Detection Analysis Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Data source: {anomaly_scores_path}\n")
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"Normal samples: {np.sum(labels == 0)}\n")
        f.write(f"Anomaly samples: {np.sum(labels == 1)}\n\n")
        
        for metric_name, scores in metrics_dict.items():
            f.write(f"\n{metric_name}:\n")
            f.write("-" * 50 + "\n")
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc_score = auc(fpr, tpr)
            f.write(f"  AUROC: {roc_auc_score:.4f}\n")
            
            # Average Precision
            precision, recall, _ = precision_recall_curve(labels, scores)
            from sklearn.metrics import average_precision_score
            ap_score = average_precision_score(labels, scores)
            f.write(f"  Average Precision: {ap_score:.4f}\n")
            
            # Score statistics
            f.write(f"\n  Normal samples statistics:\n")
            f.write(f"    Mean: {np.mean(scores[labels == 0]):.4f}\n")
            f.write(f"    Std:  {np.std(scores[labels == 0]):.4f}\n")
            f.write(f"    Min:  {np.min(scores[labels == 0]):.4f}\n")
            f.write(f"    Max:  {np.max(scores[labels == 0]):.4f}\n")
            
            f.write(f"\n  Anomaly samples statistics:\n")
            f.write(f"    Mean: {np.mean(scores[labels == 1]):.4f}\n")
            f.write(f"    Std:  {np.std(scores[labels == 1]):.4f}\n")
            f.write(f"    Min:  {np.min(scores[labels == 1]):.4f}\n")
            f.write(f"    Max:  {np.max(scores[labels == 1]):.4f}\n")
    
    print(f"  Saved: {summary_file}")
    print(f"\nAll plots and statistics saved to: {output_dir}")
    print("="*70)


# =============================================================================
# Multi-Shot Comparison Function
# =============================================================================

def generate_shots_comparison_plot(input_dir: str, output_dir: str = None):
    """
    Generate comparison plots for multi-shot experiments from a directory containing results.
    
    Args:
        input_dir: Path to directory containing multi-shot results (should have all_shots_results.json)
                   OR direct path to all_shots_results.json file
        output_dir: Directory to save plots (if None, saves to input_dir parent)
    
    The function expects either:
        1. A directory with all_shots_results.json inside
        2. A direct path to all_shots_results.json file
    """
    input_path = Path(input_dir)
    
    # Check if input is a JSON file directly
    if input_path.suffix == '.json' and input_path.exists():
        all_shots_file = input_path
        base_dir = input_path.parent
        print(f"Loading results from JSON file: {all_shots_file}")
    elif input_path.is_dir():
        # It's a directory, look for all_shots_results.json inside
        all_shots_file = input_path / "all_shots_results.json"
        base_dir = input_path
        if not all_shots_file.exists():
            print(f"Error: No all_shots_results.json found in {input_path}")
            return
    else:
        print(f"Error: Invalid input path: {input_dir}")
        return
    
    output_path = Path(output_dir) if output_dir else base_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the results
    with open(all_shots_file, 'r') as f:
        all_shots_metrics = json.load(f)
    
    # Convert string keys to integers for shot numbers
    all_shots_metrics = {int(k): v for k, v in all_shots_metrics.items()}
    
    # Extract categories and shots
    shots = sorted(all_shots_metrics.keys())
    categories = set()
    for shot_metrics in all_shots_metrics.values():
        categories.update(shot_metrics.keys())
    categories = sorted(list(categories))
    
    print(f"\nFound {len(shots)} shot values: {shots}")
    print(f"Found {len(categories)} categories: {categories}")
    
    metrics_to_plot = ['cosine', 'euclidean', 'knn']
    metric_labels = {'cosine': 'Cosine Similarity', 'euclidean': 'Euclidean Distance', 'knn': 'k-NN Distance'}
    
    # Plot 1: AUROC vs Shots for each category (all metrics)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for metric_idx, metric in enumerate(metrics_to_plot):
        ax = axes[metric_idx]
        
        for category in categories:
            aurocs = []
            valid_shots = []
            for n_shots in shots:
                if category in all_shots_metrics[n_shots]:
                    try:
                        aurocs.append(all_shots_metrics[n_shots][category][metric]['auroc'])
                        valid_shots.append(n_shots)
                    except KeyError:
                        continue
            
            if aurocs:
                ax.plot(valid_shots, aurocs, marker='o', label=category, linewidth=2)
        
        ax.set_xlabel('Number of Shots', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'AUROC vs Shots - {metric_labels.get(metric, metric)}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plot_path = output_path / "auroc_vs_shots_all_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")
    plt.close()
    
    # Plot 2: Average AUROC across all categories for each metric
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric in metrics_to_plot:
        avg_aurocs = []
        for n_shots in shots:
            aurocs = []
            for cat in categories:
                if cat in all_shots_metrics[n_shots]:
                    try:
                        aurocs.append(all_shots_metrics[n_shots][cat][metric]['auroc'])
                    except KeyError:
                        continue
            if aurocs:
                avg_aurocs.append(np.mean(aurocs))
            else:
                avg_aurocs.append(np.nan)
        
        ax.plot(shots, avg_aurocs, marker='o', label=metric_labels.get(metric, metric), 
                linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Shots', fontsize=12)
    ax.set_ylabel('Average AUROC', fontsize=12)
    ax.set_title('Average AUROC vs Number of Shots (All Categories)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    avg_plot_path = output_path / "average_auroc_vs_shots.png"
    plt.savefig(avg_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved average plot to {avg_plot_path}")
    plt.close()
    
    # Plot 3: Heatmap showing AUROC for each category and shot count (cosine metric)
    fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.8)))
    
    # Create data matrix
    data_matrix = np.zeros((len(categories), len(shots)))
    for i, category in enumerate(categories):
        for j, n_shots in enumerate(shots):
            if category in all_shots_metrics[n_shots]:
                try:
                    data_matrix[i, j] = all_shots_metrics[n_shots][category]['cosine']['auroc']
                except KeyError:
                    data_matrix[i, j] = np.nan
            else:
                data_matrix[i, j] = np.nan
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(shots)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(shots)
    ax.set_yticklabels(categories)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUROC (Cosine)', rotation=270, labelpad=20, fontsize=11)
    
    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(shots)):
            if not np.isnan(data_matrix[i, j]):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    ax.set_xlabel('Number of Shots', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_title('AUROC Heatmap: Categories vs Shots (Cosine Metric)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    heatmap_path = output_path / "auroc_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {heatmap_path}")
    plt.close()
    
    print(f"\nAll plots saved to {output_path}")



################################################# Plots ####################################################





############################# Compare SAM Mask with Ground Truth ##############################
# Load your data
image = cv2.imread(Path(__file__).parent.parent / "mvtec_ad/hazelnut/test/crack/007.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sam_mask = np.load(Path(__file__).parent / "results/hazelnut_sam_fewshot_1/visualizations/007_sam_mask.npy")  # Or however you get the SAM mask

# Compare and visualize
metrics = compare_sam_mask_with_groundtruth(
    sam_mask=sam_mask,
    groundtruth_mask_path=Path(__file__).parent.parent / "mvtec_ad/hazelnut/ground_truth/crack/007_mask.png",
    image=image,
    save_path=Path(__file__).parent / "Possible_Report_Visualizations/Comparison_Ground_Truth_with_SAM.png",
    title="SAM vs Ground Truth - Bottle Defect"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")

############################## Plot Anomaly Scores ##############################

input_path = Path(__file__).parent / "results/hazelnut_bottle_transistor_wood_multi_shot_5_10_15_20_25_30_nosam/all_shots_results.json"
output_path = Path(__file__).parent / "results/hazelnut_bottle_transistor_wood_multi_shot_5_10_15_20_25_30_nosam"
generate_shots_comparison_plot(input_path, output_path)

# ############################## Analyze Anomaly Detection Results ##############################
# Example usage: Analyze anomaly scores from a specific experiment
anomaly_scores_path = Path(__file__).parent / "results/transistor_nosam_fewshot_5/anomaly_scores.npz"
output_dir = Path(__file__).parent / "results/transistor_nosam_fewshot_5/analysis_plots"

plot_anomaly_scores(anomaly_scores_path, output_dir)


