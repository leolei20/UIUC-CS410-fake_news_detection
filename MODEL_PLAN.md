# Fake News Detection Model - Plan & Methodology

**Project**: UIUC CS410 Final Project - Fake News Detection  
**Dataset**: FakeNewsNet (PolitiFact subset) - 514 articles  
**Date**: October 2024  
**Authors**: Fred Hou, Yixiang Zhang, Xiaoxi Chen, Xiaoxiong Lei

---

## Table of Contents
1. [Problem Definition](#problem-definition)
2. [Dataset Overview](#dataset-overview)
3. [Model Pipeline](#model-pipeline)
4. [Methodology & Course Alignment](#methodology--course-alignment)
5. [Evaluation Strategy](#evaluation-strategy)
6. [Implementation Phases](#implementation-phases)
7. [Future Enhancements](#future-enhancements)

---

## Problem Definition

### Task
Binary text classification to distinguish fake news from real news articles.

**Input**: News article (title + body text)  
**Output**: Binary label (fake/real) + confidence score

### Formulation
- **Classification view**: Predict label ∈ {fake, real}
- **Ranking view**: Treat confidence scores as relevance scores for retrieval-style evaluation

### Why This Matters
Fake news detection is critical for:
- Maintaining information integrity online
- Protecting public discourse
- Demonstrating practical NLP applications learned in CS410

---

## Dataset Overview

### Current Data
- **Source**: `data/politifact_extracted.csv`
- **Size**: 514 articles
  - Fake: 241 (46.9%)
  - Real: 273 (53.1%)
- **Features**:
  - `title`: Article headline
  - `text`: Full article body
  - `clean_text`: Preprocessed text (stopwords removed, lowercase)
  - `author`: Article author (56% missing)
  - `label`: Ground truth (fake/real)

### Key Observations from EDA
| Metric | Fake News | Real News |
|--------|-----------|-----------|
| Avg Text Length | 2,339 chars | 14,429 chars |
| Avg Word Count | ~350 words | ~2,100 words |
| Avg Title Length | 11.6 words | 7.5 words |
| **Insight** | Shorter, sensational | Longer, detailed |

### Data Characteristics
- **Imbalance**: Slight (53% real / 47% fake) → use class weighting
- **Missing values**: 56% articles lack author info → won't use as primary feature
- **Text quality**: Both fake and real use political vocabulary (trump, president, obama)

---

## Model Pipeline

### Overview
```
Raw Text → Preprocessing → Feature Engineering → Model → Prediction
                                ↓
                         (TF-IDF + Style)
                                ↓
                    Logistic Regression
```

### Phase 1: Data Preparation

#### 1.1 Train/Dev/Test Split
- **Strategy**: Stratified split with fixed random seed
- **Ratio**: 70% train / 15% dev / 15% test
- **Sizes**: ~360 train / ~77 dev / ~77 test
- **Purpose**: 
  - Train: Model learning
  - Dev: Hyperparameter tuning
  - Test: Final unbiased evaluation

#### 1.2 Preprocessing (Already Done)
- Tokenization: Word-level, whitespace-based
- Normalization: Lowercase, remove URLs/special chars
- Stopword removal: English stopwords removed
- Output: `clean_text` field ready for modeling

### Phase 2: Feature Engineering

#### 2.1 Text Features (TF-IDF Vectorization)

**Body Text TF-IDF** (L3: VSM Framework)
```python
TfidfVectorizer(
    max_features=5000,        # Top 5000 terms
    ngram_range=(1, 2),       # Unigrams + bigrams
    min_df=3,                 # Term must appear in ≥3 docs
    sublinear_tf=True,        # Use log(1+tf) scaling
    norm='l2'                 # L2 document normalization
)
```

**Title TF-IDF** (Separate feature space)
```python
TfidfVectorizer(
    max_features=500,         # Smaller vocab for titles
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    norm='l2'
)
```

**Rationale**:
- TF-IDF captures term importance (L3: TF × IDF components)
- Bigrams capture phrases ("fake news", "breaking news")
- Separate title features because titles have different linguistic patterns
- Sublinear TF reduces impact of term frequency outliers

#### 2.2 Stylometric Features

From EDA, we observed clear style differences. Extract:

| Feature | Description | Hypothesis |
|---------|-------------|------------|
| `text_length` | Character count | Fake news shorter |
| `word_count` | Token count | Real news more detailed |
| `title_word_count` | Title length | Fake titles longer/clickbait |
| `punctuation_ratio` | `!?` / all chars | Fake more sensational |
| `uppercase_ratio` | Uppercase / all chars | Fake uses ALL CAPS |
| `digit_ratio` | Digits / all chars | Statistics style |
| `avg_word_length` | Mean chars per word | Vocabulary complexity |

**Normalization**: StandardScaler (mean=0, std=1)

#### 2.3 Feature Concatenation
```
Final Feature Vector = [TF-IDF_body | TF-IDF_title | style_features]
Dimension: ~5500 sparse + 7 dense features
```

### Phase 3: Model Selection

#### Baseline Model: Logistic Regression

**Choice Rationale**:
- ✅ Interpretable: Coefficients show feature importance
- ✅ Efficient: Fast training on sparse TF-IDF
- ✅ Probabilistic: Outputs calibrated probabilities
- ✅ Course-aligned: Natural extension of VSM similarity (L3-L4)

**Configuration**:
```python
LogisticRegression(
    penalty='l2',             # Ridge regularization
    C=1.0,                    # Inverse regularization (tune on dev)
    solver='saga',            # Handles L1/L2, scales well
    class_weight='balanced',  # Handle class imbalance
    max_iter=1000,
    random_state=42
)
```

**Hyperparameter Tuning** (on dev set):
- `C`: {0.1, 0.3, 1.0, 3.0, 10.0}
- Select by maximizing F1-score (macro)

---

## Methodology & Course Alignment

### Connection to CS410 Topics

| Course Topic | Application in Project |
|--------------|------------------------|
| **L1-L2: Text Retrieval** | Treat classification as ranking by probability; use retrieval metrics |
| **L3: VSM & TF-IDF** | Core feature representation; documents as term vectors |
| **L4: Word2Vec** | Future: Add pre-trained embeddings as features |
| **L6-L7: Language Models** | Future: Build unigram LMs for fake/real; score by likelihood ratio |
| **L8: Feedback** | Future: Rocchio-style feature reweighting |
| **L10: Evaluation** | Precision, Recall, F1, PR curves, AP, MAP |

### TF-IDF Formulation (L3)

For term \(t\) in document \(d\):

**Term Frequency** (sublinear):
\[
\text{tf}(t, d) = 1 + \log(\text{count}(t, d))
\]

**Inverse Document Frequency**:
\[
\text{idf}(t, D) = \log\frac{|D|}{|\{d \in D : t \in d\}|}
\]

**TF-IDF Weight**:
\[
w(t, d) = \text{tf}(t, d) \times \text{idf}(t, D)
\]

**Document Vector** (L2 normalized):
\[
\mathbf{v}_d = \frac{[w(t_1, d), w(t_2, d), \ldots, w(t_n, d)]}{\|\mathbf{w}_d\|_2}
\]

### Logistic Regression as Probabilistic Classifier

**Decision Function**:
\[
P(y=\text{fake} \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
\]

Where:
- \(\mathbf{x}\): Feature vector (TF-IDF + style)
- \(\mathbf{w}\): Learned weights
- \(b\): Bias term

**Training Objective** (with L2 regularization):
\[
\min_{\mathbf{w}, b} \left[ \sum_{i=1}^{N} \log(1 + e^{-y_i(\mathbf{w}^T \mathbf{x}_i + b)}) + \lambda \|\mathbf{w}\|_2^2 \right]
\]

Where \(\lambda = \frac{1}{C}\) (C is sklearn's inverse regularization parameter).

---

## Evaluation Strategy

### Metrics (L10: Evaluation)

#### Classification Metrics
1. **Accuracy**: Overall correctness
   \[
   \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}}
   \]

2. **Precision**: Of predicted fake, how many are actually fake?
   \[
   \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
   \]

3. **Recall**: Of actual fake, how many did we catch?
   \[
   \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
   \]

4. **F1-Score**: Harmonic mean of precision and recall
   \[
   \text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
   \]

5. **Macro-averaged**: Compute per-class, then average (treats classes equally)

#### Ranking Metrics (Treating as Retrieval)
6. **ROC-AUC**: Area under Receiver Operating Characteristic curve
   - Measures probability that random fake ranks higher than random real

7. **PR-AUC**: Area under Precision-Recall curve
   - Better for imbalanced data; focus on positive class

8. **Average Precision (AP)**: Area under PR curve (single query view)
   \[
   \text{AP} = \sum_{k=1}^{n} P(k) \cdot \Delta R(k)
   \]

### Evaluation Protocol

**Dev Set** (Hyperparameter Selection):
- Train on train set with different C values
- Evaluate all metrics on dev set
- Select C that maximizes macro-F1
- Use PR-AUC as tiebreaker

**Test Set** (Final Unbiased Results):
- Train final model on train+dev with best C
- Single evaluation on test set
- Report all metrics + visualizations
- **Never** tune on test set

### Visualizations
1. **Confusion Matrix**: Classification performance breakdown
2. **PR Curve**: Precision vs. Recall tradeoff
3. **ROC Curve**: TPR vs. FPR tradeoff
4. **Feature Importance**: Top 20 coefficients (most fake/real indicative)
5. **Error Analysis**: Sample misclassified articles

---

## Implementation Phases

### Phase 0: Setup ✓
- [x] Organize directory structure
- [x] Create `data/`, `scripts/`, `outputs/` folders
- [x] Load and explore `politifact_extracted.csv`

### Phase 1: Data Split & Persistence
- [ ] Stratified train/dev/test split (70/15/15)
- [ ] Save splits to `data/train.csv`, `data/dev.csv`, `data/test.csv`
- [ ] Verify class balance in each split

### Phase 2: Feature Engineering
- [ ] Fit TF-IDF vectorizers on train set only
- [ ] Transform train/dev/test with fitted vectorizers
- [ ] Extract stylometric features
- [ ] Concatenate feature vectors
- [ ] Save vectorizers for reproducibility

### Phase 3: Model Training
- [ ] Train Logistic Regression with different C values
- [ ] Evaluate on dev set
- [ ] Select best hyperparameters
- [ ] Train final model on train+dev

### Phase 4: Evaluation
- [ ] Generate predictions on test set
- [ ] Calculate all metrics
- [ ] Create confusion matrix
- [ ] Plot PR and ROC curves
- [ ] Analyze feature importance
- [ ] Error analysis on misclassifications

### Phase 5: Documentation
- [ ] Save results to `outputs/model_v0_results.json`
- [ ] Export visualizations
- [ ] Document findings in report

---

## Future Enhancements

### Model v1: Word Embeddings (L4-L5)
- Add averaged Word2Vec or GloVe embeddings
- Compare TF-IDF + embeddings vs. TF-IDF alone
- Ablation study to measure contribution

### Model v2: Language Model Scoring (L6-L7)
- Build unigram LMs for fake and real news
- Score documents by log-likelihood ratio
- Smoothing: Add-k or Jelinek-Mercer
- Ensemble with TF-IDF model

### Model v3: Feedback Mechanisms (L8)
- Rocchio algorithm on train set centroids
- Pseudo-relevance feedback to reweight features
- Query expansion analogy for feature engineering

### Model v4: Deep Learning
- BiLSTM or Transformer-based (BERT, RoBERTa)
- Fine-tune pre-trained models
- Compare with classical methods

### Model v5: Social Context Integration
- When FakeNewsNet social data available:
  - User credibility scores
  - Tweet/retweet counts
  - Propagation network features
- Multi-modal fusion (text + social)

---

## Expected Outcomes

### Baseline Performance Goals
Based on related work and our EDA:

| Metric | Target |
|--------|--------|
| Accuracy | > 80% |
| F1 (macro) | > 0.75 |
| PR-AUC | > 0.85 |

### Success Criteria
1. ✅ Model significantly outperforms random (50% accuracy)
2. ✅ Model outperforms majority baseline (53% accuracy)
3. ✅ F1-score balanced across both classes
4. ✅ Interpretable feature weights align with domain knowledge
5. ✅ Reproducible pipeline with fixed seeds

### Deliverables
1. **Code**: Clean, documented Python scripts
2. **Model**: Saved vectorizers and trained model
3. **Results**: Metrics, curves, error analysis
4. **Report**: Findings, methodology, future work
5. **Presentation**: Video demonstration

---

## References

1. Shu, K., et al. (2018). "FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media." arXiv:1809.01286
2. CS410 Course Materials: Lectures 1-10 (Text Retrieval, VSM, TF-IDF, Evaluation)
3. Scikit-learn Documentation: TfidfVectorizer, LogisticRegression

---

## Appendix: File Structure

```
UIUC-CS410-fake_news_detection/
├── data/
│   ├── politifact_extracted.csv    # Original data
│   ├── train.csv                   # Training split (to be created)
│   ├── dev.csv                     # Development split
│   └── test.csv                    # Test split
├── scripts/
│   ├── preprocessing.py            # Original preprocessing
│   ├── data_analysis.py            # EDA script
│   ├── train_baseline.py           # Model training (to be created)
│   └── evaluate.py                 # Evaluation script (to be created)
├── outputs/
│   ├── vectorizers/                # Saved TF-IDF vectorizers
│   ├── models/                     # Saved trained models
│   ├── results/                    # Metrics and predictions
│   └── figures/                    # Visualizations
├── MODEL_PLAN.md                   # This file
├── README.md                       # Project overview
└── requirements.txt                # Dependencies
```

---

**Document Version**: 1.0  
**Last Updated**: October 22, 2024  
**Status**: Ready for Implementation

