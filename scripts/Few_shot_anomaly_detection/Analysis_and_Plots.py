import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


data = np.load("C:\\Users\\riang\\OneDrive\\Dokumente\\ETH\\Master\\DTU 1. Sem\\Deep Learning\\AnomalyDetection\\scripts\\Few_shot_anomaly_detection\\results\\carpet_fewshot_10\\anomaly_scores.npz")

print(data.files)     # list keys
print("Shots:", data["n_shots"])
print("Seed:", data["seed"])

# Load all three metrics
scores_cosine = data["scores_cosine"]
scores_euclidean = data["scores_euclidean"]
scores_knn = data["scores_knn"]
labels = data["labels"]

print(f"Cosine scores shape: {scores_cosine.shape}")
print(f"Euclidean scores shape: {scores_euclidean.shape}")
print(f"k-NN scores shape: {scores_knn.shape}")
print(f"Labels shape: {labels.shape}")

metrics_dict = {
    'Cosine Similarity': scores_cosine,
    'Euclidean Distance': scores_euclidean,
    'k-NN Distance': scores_knn
}

# 1. Score Distribution (Histograms)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (metric_name, scores) in enumerate(metrics_dict.items()):
    axes[idx].hist(scores[labels == 0], bins=10, alpha=0.7, label="Normal")
    axes[idx].hist(scores[labels == 1], bins=30, alpha=0.7, label="Anomaly")
    axes[idx].legend()
    axes[idx].set_xlabel("Anomaly Score")
    axes[idx].set_ylabel("Count")
    axes[idx].set_title(f"Score Distribution - {metric_name}")
plt.tight_layout()
plt.show()

# 2. Density Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (metric_name, scores) in enumerate(metrics_dict.items()):
    sns.kdeplot(scores[labels == 0], shade=True, label="Normal", ax=axes[idx])
    sns.kdeplot(scores[labels == 1], shade=True, label="Anomaly", ax=axes[idx])
    axes[idx].set_xlabel("Anomaly Score")
    axes[idx].set_title(f"Score Density - {metric_name}")
    axes[idx].legend()
plt.tight_layout()
plt.show()

# 3. Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (metric_name, scores) in enumerate(metrics_dict.items()):
    axes[idx].scatter(range(len(scores)), scores, c=labels, cmap="coolwarm", alpha=0.6)
    axes[idx].set_xlabel("Sample Index")
    axes[idx].set_ylabel("Anomaly Score")
    axes[idx].set_title(f"Scores per Sample - {metric_name}")
plt.tight_layout()
plt.show()

# 4. ROC Curves (all on one plot for comparison)
plt.figure(figsize=(8, 6))
for metric_name, scores in metrics_dict.items():
    fpr, tpr, _ = roc_curve(labels, scores) 
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{metric_name} (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0,1], [0,1], "--", color='gray', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Metrics")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 5. Precision-Recall Curves (all on one plot for comparison)
plt.figure(figsize=(8, 6))
for metric_name, scores in metrics_dict.items():
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.plot(recall, precision, label=metric_name, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - All Metrics")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

