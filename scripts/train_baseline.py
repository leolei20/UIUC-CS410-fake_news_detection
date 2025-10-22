"""
Baseline Fake News Detection Model
===================================
TF-IDF + Logistic Regression

Course Alignment: CS410 L3 (VSM, TF-IDF), L10 (Evaluation)
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("FAKE NEWS DETECTION - BASELINE MODEL")
print("="*80)
print("\nModel: TF-IDF + Logistic Regression")
print("Features: Body text + Title + Stylometric features")
print()

# ============================================================================
# STEP 1: LOAD DATA AND CREATE SPLITS
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING AND SPLITTING")
print("="*80)

# Load data
df = pd.read_csv('../data/politifact_extracted.csv')
print(f"\nTotal records: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Remove rows with missing text
df = df.dropna(subset=['clean_text', 'title'])
print(f"\nRecords after removing missing text: {len(df)}")

# Prepare features and labels
X = df[['clean_text', 'title', 'text']].copy()
y = (df['label'] == 'fake').astype(int)  # 1 for fake, 0 for real

print(f"\nClass distribution:")
print(f"  Fake (1): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Real (0): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

# Split data: 70% train, 15% dev, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=RANDOM_SEED, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Dev:   {len(X_dev)} ({len(X_dev)/len(df)*100:.1f}%)")
print(f"  Test:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

# Verify stratification
print(f"\nClass balance check:")
print(f"  Train - Fake: {y_train.mean()*100:.1f}%")
print(f"  Dev   - Fake: {y_dev.mean()*100:.1f}%")
print(f"  Test  - Fake: {y_test.mean()*100:.1f}%")

# Save splits for future use
train_df = X_train.copy()
train_df['label'] = y_train.values
train_df.to_csv('../data/train.csv', index=False)

dev_df = X_dev.copy()
dev_df['label'] = y_dev.values
dev_df.to_csv('../data/dev.csv', index=False)

test_df = X_test.copy()
test_df['label'] = y_test.values
test_df.to_csv('../data/test.csv', index=False)

print("\n[OK] Data splits saved to data/ folder")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING")
print("="*80)

# 2.1: TF-IDF Features
print("\n[2.1] TF-IDF Vectorization...")

# Body text TF-IDF
print("  - Fitting TF-IDF on body text (clean_text)...")
tfidf_body = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    norm='l2'
)
X_train_body_tfidf = tfidf_body.fit_transform(X_train['clean_text'])
X_dev_body_tfidf = tfidf_body.transform(X_dev['clean_text'])
X_test_body_tfidf = tfidf_body.transform(X_test['clean_text'])
print(f"    Body TF-IDF shape: {X_train_body_tfidf.shape}")

# Title TF-IDF
print("  - Fitting TF-IDF on titles...")
tfidf_title = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    norm='l2'
)
X_train_title_tfidf = tfidf_title.fit_transform(X_train['title'].fillna(''))
X_dev_title_tfidf = tfidf_title.transform(X_dev['title'].fillna(''))
X_test_title_tfidf = tfidf_title.transform(X_test['title'].fillna(''))
print(f"    Title TF-IDF shape: {X_train_title_tfidf.shape}")

# 2.2: Stylometric Features
print("\n[2.2] Extracting stylometric features...")

def extract_style_features(df):
    """Extract style-based features from text"""
    features = pd.DataFrame()
    
    # Length features
    features['text_length'] = df['text'].astype(str).str.len()
    features['word_count'] = df['text'].astype(str).str.split().str.len()
    features['title_word_count'] = df['title'].astype(str).str.split().str.len()
    
    # Character-based ratios
    text = df['text'].astype(str)
    features['punctuation_ratio'] = text.str.count(r'[!?]') / features['text_length']
    features['uppercase_ratio'] = text.str.count(r'[A-Z]') / features['text_length']
    features['digit_ratio'] = text.str.count(r'\d') / features['text_length']
    
    # Word complexity
    features['avg_word_length'] = features['text_length'] / features['word_count']
    
    # Handle infinities and NaNs
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    
    return features

X_train_style = extract_style_features(X_train)
X_dev_style = extract_style_features(X_dev)
X_test_style = extract_style_features(X_test)

print(f"    Style features shape: {X_train_style.shape}")
print(f"    Features: {list(X_train_style.columns)}")

# Normalize style features
print("  - Normalizing style features...")
scaler = StandardScaler()
X_train_style_scaled = scaler.fit_transform(X_train_style)
X_dev_style_scaled = scaler.transform(X_dev_style)
X_test_style_scaled = scaler.transform(X_test_style)

# 2.3: Concatenate all features
print("\n[2.3] Concatenating feature vectors...")
from scipy.sparse import hstack, csr_matrix

X_train_final = hstack([
    X_train_body_tfidf,
    X_train_title_tfidf,
    csr_matrix(X_train_style_scaled)
])
X_dev_final = hstack([
    X_dev_body_tfidf,
    X_dev_title_tfidf,
    csr_matrix(X_dev_style_scaled)
])
X_test_final = hstack([
    X_test_body_tfidf,
    X_test_title_tfidf,
    csr_matrix(X_test_style_scaled)
])

print(f"    Final feature vector shape: {X_train_final.shape}")
print(f"    (Body TF-IDF: {X_train_body_tfidf.shape[1]} + "
      f"Title TF-IDF: {X_train_title_tfidf.shape[1]} + "
      f"Style: {X_train_style.shape[1]})")

# Save vectorizers and scaler
print("\n[OK] Saving feature extractors...")
output_dir = Path('../outputs')
vectorizer_dir = output_dir / 'vectorizers'
vectorizer_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(tfidf_body, vectorizer_dir / 'tfidf_body.pkl')
joblib.dump(tfidf_title, vectorizer_dir / 'tfidf_title.pkl')
joblib.dump(scaler, vectorizer_dir / 'style_scaler.pkl')
print("    Saved: tfidf_body.pkl, tfidf_title.pkl, style_scaler.pkl")

# ============================================================================
# STEP 3: MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: MODEL TRAINING")
print("="*80)

# Hyperparameter search on dev set
C_values = [0.1, 0.3, 1.0, 3.0, 10.0]
print(f"\n[3.1] Hyperparameter tuning on dev set...")
print(f"      Testing C values: {C_values}")

best_c = None
best_f1 = 0
results = []

for c in C_values:
    print(f"\n  Training with C={c}...")
    model = LogisticRegression(
        penalty='l2',
        C=c,
        solver='saga',
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train_final, y_train)
    
    # Evaluate on dev set
    y_dev_pred = model.predict(X_dev_final)
    y_dev_proba = model.predict_proba(X_dev_final)[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_dev, y_dev_pred, average='macro'
    )
    accuracy = accuracy_score(y_dev, y_dev_pred)
    pr_auc = average_precision_score(y_dev, y_dev_proba)
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Macro-F1: {f1:.4f}")
    print(f"    PR-AUC:   {pr_auc:.4f}")
    
    results.append({
        'C': c,
        'accuracy': accuracy,
        'macro_f1': f1,
        'pr_auc': pr_auc
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_c = c

print(f"\n[OK] Best hyperparameter: C={best_c} (Macro-F1: {best_f1:.4f})")

# Train final model with best hyperparameter
print(f"\n[3.2] Training final model with C={best_c}...")
final_model = LogisticRegression(
    penalty='l2',
    C=best_c,
    solver='saga',
    class_weight='balanced',
    max_iter=1000,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
final_model.fit(X_train_final, y_train)

# Save model
model_dir = output_dir / 'models'
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, model_dir / 'baseline_model.pkl')
print(f"    Saved: baseline_model.pkl")

# ============================================================================
# STEP 4: EVALUATION ON TEST SET
# ============================================================================
print("\n" + "="*80)
print("STEP 4: FINAL EVALUATION ON TEST SET")
print("="*80)

# Predictions
y_test_pred = final_model.predict(X_test_final)
y_test_proba = final_model.predict_proba(X_test_final)[:, 1]

# Classification metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average='macro'
)
precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
    y_test, y_test_pred, average=None
)

# Ranking metrics
roc_auc = roc_auc_score(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)

print("\n[4.1] Classification Metrics:")
print(f"  Accuracy:       {accuracy:.4f}")
print(f"  Macro-F1:       {f1_macro:.4f}")
print(f"  Macro-Precision:{precision_macro:.4f}")
print(f"  Macro-Recall:   {recall_macro:.4f}")

print("\n[4.2] Per-Class Metrics:")
print(f"  Real (0):")
print(f"    Precision: {precision_per_class[0]:.4f}")
print(f"    Recall:    {recall_per_class[0]:.4f}")
print(f"    F1-Score:  {f1_per_class[0]:.4f}")
print(f"    Support:   {support[0]}")
print(f"  Fake (1):")
print(f"    Precision: {precision_per_class[1]:.4f}")
print(f"    Recall:    {recall_per_class[1]:.4f}")
print(f"    F1-Score:  {f1_per_class[1]:.4f}")
print(f"    Support:   {support[1]}")

print("\n[4.3] Ranking Metrics:")
print(f"  ROC-AUC:        {roc_auc:.4f}")
print(f"  PR-AUC (AP):    {pr_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\n[4.4] Confusion Matrix:")
print("              Predicted")
print("              Real  Fake")
print(f"Actual Real   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Fake   {cm[1,0]:4d}  {cm[1,1]:4d}")

# Save results
results_dict = {
    'model': 'Logistic Regression (TF-IDF + Style)',
    'hyperparameters': {'C': best_c, 'class_weight': 'balanced'},
    'test_metrics': {
        'accuracy': float(accuracy),
        'macro_f1': float(f1_macro),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    },
    'per_class_metrics': {
        'real': {
            'precision': float(precision_per_class[0]),
            'recall': float(recall_per_class[0]),
            'f1': float(f1_per_class[0]),
            'support': int(support[0])
        },
        'fake': {
            'precision': float(precision_per_class[1]),
            'recall': float(recall_per_class[1]),
            'f1': float(f1_per_class[1]),
            'support': int(support[1])
        }
    },
    'confusion_matrix': cm.tolist(),
    'hyperparameter_search': results
}

results_dir = output_dir / 'results'
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / 'baseline_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"\n[OK] Results saved to outputs/results/baseline_results.json")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nNext: Run evaluate.py for visualizations and error analysis")

