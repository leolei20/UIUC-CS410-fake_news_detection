"""
DistilBERT Model Evaluation and Visualization
==============================================
Generate visualizations and compare with baseline model
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DISTILBERT MODEL EVALUATION & VISUALIZATION")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA AND RESULTS
# ============================================================================
print("\n[1] Loading test data and predictions...")

base_dir = Path(__file__).parent
data_dir = base_dir / ".." / "data"
results_dir = base_dir / ".." / "outputs" / "results"
figures_dir = base_dir / ".." / "outputs" / "figures"

figures_dir.mkdir(parents=True, exist_ok=True)

# Load test data
test_df = pd.read_csv(data_dir / "test.csv")
y_test = test_df['label'].values

# Load BERT predictions
bert_pred_df = pd.read_csv(results_dir / "distilbert_test_predictions.csv")
y_bert_pred = bert_pred_df['predicted'].values
y_bert_proba = bert_pred_df['probability_fake'].values

# Load baseline predictions for comparison
baseline_pred_df = pd.read_csv(results_dir / "test_predictions.csv")
y_baseline_pred = baseline_pred_df['predicted'].values
y_baseline_proba = baseline_pred_df['probability_fake'].values

# Load metrics
with open(results_dir / "distilbert_results.json", "r") as f:
    bert_results = json.load(f)
with open(results_dir / "baseline_results.json", "r") as f:
    baseline_results = json.load(f)

print(f"    Test set size: {len(test_df)}")
print(f"    BERT predictions loaded")
print(f"    Baseline predictions loaded")

# ============================================================================
# VISUALIZATION 1: CONFUSION MATRICES COMPARISON
# ============================================================================
print("\n[2] Creating visualizations...")
print("    [2.1] Confusion matrices comparison...")

cm_bert = confusion_matrix(y_test, y_bert_pred)
cm_baseline = confusion_matrix(y_test, y_baseline_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline confusion matrix
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            ax=axes[0])
axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[0].set_title(f'Baseline Model\nAccuracy: {baseline_results["test_metrics"]["accuracy"]:.1%}', 
                  fontsize=12, fontweight='bold')

# BERT confusion matrix
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            ax=axes[1])
axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[1].set_title(f'DistilBERT Model\nAccuracy: {bert_results["test_metrics"]["accuracy"]:.1%}', 
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'bert_confusion_comparison.png', dpi=300, bbox_inches='tight')
print(f"          Saved: bert_confusion_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: ROC CURVES COMPARISON
# ============================================================================
print("    [2.2] ROC curves comparison...")

fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_baseline_proba)
fpr_bert, tpr_bert, _ = roc_curve(y_test, y_bert_proba)

roc_auc_baseline = roc_auc_score(y_test, y_baseline_proba)
roc_auc_bert = roc_auc_score(y_test, y_bert_proba)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(fpr_baseline, tpr_baseline, linewidth=2.5, 
        label=f'Baseline (AUC={roc_auc_baseline:.3f})', color='#3498db')
ax.plot(fpr_bert, tpr_bert, linewidth=2.5, 
        label=f'DistilBERT (AUC={roc_auc_bert:.3f})', color='#2ecc71')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random', alpha=0.5)

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(figures_dir / 'bert_roc_comparison.png', dpi=300, bbox_inches='tight')
print(f"          Saved: bert_roc_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: PR CURVES COMPARISON
# ============================================================================
print("    [2.3] Precision-Recall curves comparison...")

precision_baseline, recall_baseline, _ = precision_recall_curve(y_test, y_baseline_proba)
precision_bert, recall_bert, _ = precision_recall_curve(y_test, y_bert_proba)

pr_auc_baseline = average_precision_score(y_test, y_baseline_proba)
pr_auc_bert = average_precision_score(y_test, y_bert_proba)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(recall_baseline, precision_baseline, linewidth=2.5,
        label=f'Baseline (AP={pr_auc_baseline:.3f})', color='#3498db')
ax.plot(recall_bert, precision_bert, linewidth=2.5,
        label=f'DistilBERT (AP={pr_auc_bert:.3f})', color='#2ecc71')
ax.plot([0, 1], [y_test.mean(), y_test.mean()], 'k--', linewidth=1.5, 
        label='Random', alpha=0.5)

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(figures_dir / 'bert_pr_comparison.png', dpi=300, bbox_inches='tight')
print(f"          Saved: bert_pr_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: METRICS COMPARISON BAR CHART
# ============================================================================
print("    [2.4] Metrics comparison chart...")

metrics_data = {
    'Accuracy': [
        baseline_results['test_metrics']['accuracy'],
        bert_results['test_metrics']['accuracy']
    ],
    'Macro-F1': [
        baseline_results['test_metrics']['macro_f1'],
        bert_results['test_metrics']['macro_f1']
    ],
    'ROC-AUC': [
        baseline_results['test_metrics']['roc_auc'],
        bert_results['test_metrics']['roc_auc']
    ],
    'PR-AUC': [
        baseline_results['test_metrics']['pr_auc'],
        bert_results['test_metrics']['pr_auc']
    ]
}

x = np.arange(len(metrics_data))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, [v[0] for v in metrics_data.values()], width,
               label='Baseline', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, [v[1] for v in metrics_data.values()], width,
               label='DistilBERT', color='#2ecc71', alpha=0.8)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_data.keys(), fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0.75, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.savefig(figures_dir / 'bert_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"          Saved: bert_metrics_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: ERROR REDUCTION
# ============================================================================
print("    [2.5] Error analysis comparison...")

baseline_errors = (y_test != y_baseline_pred).sum()
bert_errors = (y_test != y_bert_pred).sum()

fig, ax = plt.subplots(figsize=(10, 6))

models = ['Baseline', 'DistilBERT']
correct = [len(y_test) - baseline_errors, len(y_test) - bert_errors]
errors = [baseline_errors, bert_errors]

x = np.arange(len(models))
width = 0.6

bars1 = ax.bar(x, correct, width, label='Correct', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x, errors, width, bottom=correct, label='Errors', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
ax.set_title(f'Prediction Accuracy on Test Set (n={len(y_test)})', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, len(y_test) + 5])

# Add labels
for i, (c, e) in enumerate(zip(correct, errors)):
    ax.text(i, c/2, f'{c}\n({c/len(y_test)*100:.1f}%)', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(i, c + e/2, f'{e}\n({e/len(y_test)*100:.1f}%)', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig(figures_dir / 'bert_error_comparison.png', dpi=300, bbox_inches='tight')
print(f"          Saved: bert_error_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: PROBABILITY DISTRIBUTIONS
# ============================================================================
print("    [2.6] Prediction probability distributions...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Baseline
axes[0].hist(y_baseline_proba[y_test == 0], bins=30, alpha=0.6, 
             label='Real News', color='#3498db')
axes[0].hist(y_baseline_proba[y_test == 1], bins=30, alpha=0.6, 
             label='Fake News', color='#e74c3c')
axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, 
                label='Decision Threshold')
axes[0].set_xlabel('Predicted Probability (Fake)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[0].set_title('Baseline Model - Probability Distribution', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# BERT
axes[1].hist(y_bert_proba[y_test == 0], bins=30, alpha=0.6, 
             label='Real News', color='#3498db')
axes[1].hist(y_bert_proba[y_test == 1], bins=30, alpha=0.6, 
             label='Fake News', color='#e74c3c')
axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, 
                label='Decision Threshold')
axes[1].set_xlabel('Predicted Probability (Fake)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1].set_title('DistilBERT Model - Probability Distribution', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'bert_probability_distributions.png', dpi=300, bbox_inches='tight')
print(f"          Saved: bert_probability_distributions.png")
plt.close()

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
print("\n[3] Detailed Error Analysis...")

# Find where models differ
baseline_wrong = (y_test != y_baseline_pred)
bert_wrong = (y_test != y_bert_pred)

both_wrong = baseline_wrong & bert_wrong
only_baseline_wrong = baseline_wrong & ~bert_wrong
only_bert_wrong = ~baseline_wrong & bert_wrong
both_right = ~baseline_wrong & ~bert_wrong

print(f"\n    Both models correct:      {both_right.sum()}/{len(y_test)} ({both_right.mean()*100:.1f}%)")
print(f"    Only baseline wrong:      {only_baseline_wrong.sum()} (BERT fixed these!)")
print(f"    Only BERT wrong:          {only_bert_wrong.sum()} (BERT made new errors)")
print(f"    Both models wrong:        {both_wrong.sum()} (Hard cases)")

# Examples where BERT fixed baseline errors
print(f"\n    [3.1] Examples where DistilBERT FIXED baseline errors:")
fixed_indices = np.where(only_baseline_wrong)[0]
if len(fixed_indices) > 0:
    for i, idx in enumerate(fixed_indices[:3], 1):
        title = str(test_df.iloc[idx]['title']).encode('ascii', errors='ignore').decode('ascii')
        title = title[:70] + "..." if len(title) > 70 else title
        true_label = "FAKE" if y_test[idx] == 1 else "REAL"
        baseline_conf = y_baseline_proba[idx]
        bert_conf = y_bert_proba[idx]
        print(f"        {i}. True: {true_label}")
        print(f"           Title: {title}")
        print(f"           Baseline: {baseline_conf:.3f} (WRONG), BERT: {bert_conf:.3f} (CORRECT)")
else:
    print("        None!")

# Examples where BERT made new errors
print(f"\n    [3.2] Examples where DistilBERT made NEW errors:")
new_error_indices = np.where(only_bert_wrong)[0]
if len(new_error_indices) > 0:
    for i, idx in enumerate(new_error_indices[:3], 1):
        title = str(test_df.iloc[idx]['title']).encode('ascii', errors='ignore').decode('ascii')
        title = title[:70] + "..." if len(title) > 70 else title
        true_label = "FAKE" if y_test[idx] == 1 else "REAL"
        baseline_conf = y_baseline_proba[idx]
        bert_conf = y_bert_proba[idx]
        print(f"        {i}. True: {true_label}")
        print(f"           Title: {title}")
        print(f"           Baseline: {baseline_conf:.3f} (CORRECT), BERT: {bert_conf:.3f} (WRONG)")
else:
    print("        None!")

# Hard cases - both wrong
print(f"\n    [3.3] Hard cases (both models WRONG):")
hard_indices = np.where(both_wrong)[0]
if len(hard_indices) > 0:
    for i, idx in enumerate(hard_indices[:3], 1):
        title = str(test_df.iloc[idx]['title']).encode('ascii', errors='ignore').decode('ascii')
        title = title[:70] + "..." if len(title) > 70 else title
        true_label = "FAKE" if y_test[idx] == 1 else "REAL"
        baseline_conf = y_baseline_proba[idx]
        bert_conf = y_bert_proba[idx]
        print(f"        {i}. True: {true_label}")
        print(f"           Title: {title}")
        print(f"           Baseline: {baseline_conf:.3f} (WRONG), BERT: {bert_conf:.3f} (WRONG)")
else:
    print("        None! All errors are unique to one model.")

# Save comparison report
comparison_report = {
    "summary": {
        "both_correct": int(both_right.sum()),
        "only_baseline_wrong": int(only_baseline_wrong.sum()),
        "only_bert_wrong": int(only_bert_wrong.sum()),
        "both_wrong": int(both_wrong.sum()),
        "improvement": int(only_baseline_wrong.sum() - only_bert_wrong.sum())
    },
    "baseline_metrics": baseline_results['test_metrics'],
    "bert_metrics": bert_results['test_metrics']
}

with open(results_dir / 'model_comparison.json', 'w') as f:
    json.dump(comparison_report, f, indent=2)

print(f"\n    [OK] Comparison report saved to: model_comparison.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print("\nGenerated visualizations in outputs/figures/:")
print("  1. bert_confusion_comparison.png      - Side-by-side confusion matrices")
print("  2. bert_roc_comparison.png            - ROC curves overlay")
print("  3. bert_pr_comparison.png             - PR curves overlay")
print("  4. bert_metrics_comparison.png        - Bar chart of all metrics")
print("  5. bert_error_comparison.png          - Error reduction visualization")
print("  6. bert_probability_distributions.png - Confidence distributions")
print("\nComparison report:")
print("  - model_comparison.json")
print("\n" + "="*80)

