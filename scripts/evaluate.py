"""
Model Evaluation and Visualization
===================================
Generate visualizations and error analysis for the baseline model
"""

import pandas as pd
import numpy as np
import json
import joblib
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
print("MODEL EVALUATION & VISUALIZATION")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================
print("\n[1] Loading test data and model...")

# Load test data
test_df = pd.read_csv('../data/test.csv')
X_test = test_df[['clean_text', 'title', 'text']].copy()
y_test = test_df['label'].values

print(f"    Test set size: {len(test_df)}")

# Load vectorizers and model
vectorizer_dir = Path('../outputs/vectorizers')
model_dir = Path('../outputs/models')

tfidf_body = joblib.load(vectorizer_dir / 'tfidf_body.pkl')
tfidf_title = joblib.load(vectorizer_dir / 'tfidf_title.pkl')
scaler = joblib.load(vectorizer_dir / 'style_scaler.pkl')
model = joblib.load(model_dir / 'baseline_model.pkl')

print("    [OK] Loaded vectorizers and model")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
print("\n[2] Extracting features...")

# TF-IDF
X_test_body = tfidf_body.transform(X_test['clean_text'])
X_test_title = tfidf_title.transform(X_test['title'].fillna(''))

# Style features
def extract_style_features(df):
    features = pd.DataFrame()
    features['text_length'] = df['text'].astype(str).str.len()
    features['word_count'] = df['text'].astype(str).str.split().str.len()
    features['title_word_count'] = df['title'].astype(str).str.split().str.len()
    text = df['text'].astype(str)
    features['punctuation_ratio'] = text.str.count(r'[!?]') / features['text_length']
    features['uppercase_ratio'] = text.str.count(r'[A-Z]') / features['text_length']
    features['digit_ratio'] = text.str.count(r'\d') / features['text_length']
    features['avg_word_length'] = features['text_length'] / features['word_count']
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    return features

X_test_style = extract_style_features(X_test)
X_test_style_scaled = scaler.transform(X_test_style)

# Concatenate
from scipy.sparse import hstack, csr_matrix
X_test_final = hstack([X_test_body, X_test_title, csr_matrix(X_test_style_scaled)])

print(f"    Feature vector shape: {X_test_final.shape}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n[3] Generating predictions...")

y_pred = model.predict(X_test_final)
y_proba = model.predict_proba(X_test_final)[:, 1]

print(f"    Predictions generated: {len(y_pred)}")

# ============================================================================
# VISUALIZATION 1: CONFUSION MATRIX
# ============================================================================
print("\n[4] Creating visualizations...")
print("    [4.1] Confusion matrix...")

figures_dir = Path('../outputs/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Baseline Model', fontsize=14, fontweight='bold')

# Add percentage annotations
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm[i].sum() * 100
        ax.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(figures_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"          Saved: confusion_matrix.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: PR CURVE
# ============================================================================
print("    [4.2] Precision-Recall curve...")

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, linewidth=2, label=f'Baseline (AP={pr_auc:.3f})')
ax.plot([0, 1], [y_test.mean(), y_test.mean()], 
        'k--', linewidth=1, label='Random')
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(figures_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
print(f"          Saved: pr_curve.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: ROC CURVE
# ============================================================================
print("    [4.3] ROC curve...")

fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'Baseline (AUC={roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(figures_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
print(f"          Saved: roc_curve.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: FEATURE IMPORTANCE
# ============================================================================
print("    [4.4] Feature importance...")

# Get feature names
body_features = tfidf_body.get_feature_names_out()
title_features = [f"title_{f}" for f in tfidf_title.get_feature_names_out()]
style_features = ['text_length', 'word_count', 'title_word_count', 
                  'punctuation_ratio', 'uppercase_ratio', 'digit_ratio', 'avg_word_length']
all_features = list(body_features) + list(title_features) + style_features

# Get coefficients
coefficients = model.coef_[0]

# Top features for FAKE (positive coefficients)
top_fake_idx = np.argsort(coefficients)[-20:][::-1]
top_fake_features = [all_features[i] for i in top_fake_idx]
top_fake_coefs = coefficients[top_fake_idx]

# Top features for REAL (negative coefficients)
top_real_idx = np.argsort(coefficients)[:20]
top_real_features = [all_features[i] for i in top_real_idx]
top_real_coefs = coefficients[top_real_idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Fake indicators
axes[0].barh(range(len(top_fake_features)), top_fake_coefs, color='#ff6b6b')
axes[0].set_yticks(range(len(top_fake_features)))
axes[0].set_yticklabels(top_fake_features, fontsize=9)
axes[0].set_xlabel('Coefficient (Positive = Fake)', fontsize=11, fontweight='bold')
axes[0].set_title('Top 20 Features Indicating FAKE News', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Real indicators
axes[1].barh(range(len(top_real_features)), top_real_coefs, color='#4ecdc4')
axes[1].set_yticks(range(len(top_real_features)))
axes[1].set_yticklabels(top_real_features, fontsize=9)
axes[1].set_xlabel('Coefficient (Negative = Real)', fontsize=11, fontweight='bold')
axes[1].set_title('Top 20 Features Indicating REAL News', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print(f"          Saved: feature_importance.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: PREDICTION DISTRIBUTION
# ============================================================================
print("    [4.5] Prediction probability distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot distributions
ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6, label='Real News', color='#4ecdc4')
ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6, label='Fake News', color='#ff6b6b')
ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')

ax.set_xlabel('Predicted Probability (Fake)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
print(f"          Saved: probability_distribution.png")
plt.close()

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
print("\n[5] Error Analysis...")

# Create results dataframe
results_df = test_df.copy()
results_df['predicted'] = y_pred
results_df['probability_fake'] = y_proba
results_df['correct'] = (y_pred == y_test)

# Identify errors
false_positives = results_df[(results_df['label'] == 0) & (results_df['predicted'] == 1)]
false_negatives = results_df[(results_df['label'] == 1) & (results_df['predicted'] == 0)]

print(f"\n    False Positives (Real predicted as Fake): {len(false_positives)}")
print(f"    False Negatives (Fake predicted as Real): {len(false_negatives)}")

# Top confident errors
print("\n    [5.1] Most confident FALSE POSITIVES:")
if len(false_positives) > 0:
    top_fp = false_positives.nlargest(5, 'probability_fake')
    for idx, (i, row) in enumerate(top_fp.iterrows(), 1):
        title = str(row['title']).encode('ascii', errors='ignore').decode('ascii')
        title = title[:80] + "..." if len(title) > 80 else title
        print(f"        {idx}. [{row['probability_fake']:.3f}] {title}")
else:
    print("        None!")

print("\n    [5.2] Most confident FALSE NEGATIVES:")
if len(false_negatives) > 0:
    top_fn = false_negatives.nsmallest(5, 'probability_fake')
    for idx, (i, row) in enumerate(top_fn.iterrows(), 1):
        title = str(row['title']).encode('ascii', errors='ignore').decode('ascii')
        title = title[:80] + "..." if len(title) > 80 else title
        print(f"        {idx}. [{row['probability_fake']:.3f}] {title}")
else:
    print("        None!")

# Save error analysis
error_analysis = {
    'false_positives': {
        'count': len(false_positives),
        'examples': [
            {
                'title': row['title'],
                'probability': float(row['probability_fake']),
                'text_preview': str(row['text'])[:200] + "..."
            }
            for _, row in false_positives.nlargest(10, 'probability_fake').iterrows()
        ]
    },
    'false_negatives': {
        'count': len(false_negatives),
        'examples': [
            {
                'title': row['title'],
                'probability': float(row['probability_fake']),
                'text_preview': str(row['text'])[:200] + "..."
            }
            for _, row in false_negatives.nsmallest(10, 'probability_fake').iterrows()
        ]
    }
}

results_dir = Path('../outputs/results')
with open(results_dir / 'error_analysis.json', 'w') as f:
    json.dump(error_analysis, f, indent=2)

# Save full predictions
results_df.to_csv(results_dir / 'test_predictions.csv', index=False)

print(f"\n    [OK] Error analysis saved to outputs/results/")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  outputs/figures/")
print("    - confusion_matrix.png")
print("    - pr_curve.png")
print("    - roc_curve.png")
print("    - feature_importance.png")
print("    - probability_distribution.png")
print("  outputs/results/")
print("    - baseline_results.json")
print("    - error_analysis.json")
print("    - test_predictions.csv")
print("\n" + "="*80)

