# Baseline Model Results Summary

**Date**: October 22, 2024  
**Model**: Logistic Regression with TF-IDF + Stylometric Features  
**Dataset**: FakeNewsNet (PolitiFact subset) - 504 articles  

---

## Executive Summary

✅ **Mission Accomplished**: Successfully built and evaluated a baseline fake news detection model that significantly outperforms random and majority baselines.

### Key Performance Metrics (Test Set)
| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | **86.84%** | ✅ Exceeds 80% target |
| **Macro-F1** | **0.8683** | ✅ Exceeds 0.75 target |
| **ROC-AUC** | **0.9257** | ✅ Excellent discrimination |
| **PR-AUC** | **0.8980** | ✅ Exceeds 0.85 target |

**Interpretation**: The model correctly identifies fake news **87% of the time**, with balanced performance across both classes.

---

## Model Architecture

### Features (Total: 5,507 dimensions)

#### 1. TF-IDF Body Text (5,000 features)
- **Algorithm**: Term Frequency-Inverse Document Frequency
- **N-grams**: Unigrams + Bigrams (1-2)
- **Min Document Frequency**: 3
- **Transformation**: Sublinear TF (log-scaled)
- **Normalization**: L2 norm
- **Purpose**: Capture semantic content and key terms

#### 2. TF-IDF Title (500 features)
- **Algorithm**: Same as body, but smaller vocabulary
- **Min Document Frequency**: 2
- **Purpose**: Capture headline patterns (clickbait vs. professional)

#### 3. Stylometric Features (7 features)
- `text_length`: Total character count
- `word_count`: Total word count  
- `title_word_count`: Title length
- `punctuation_ratio`: Exclamation/question marks frequency
- `uppercase_ratio`: ALL CAPS usage
- `digit_ratio`: Numeric character frequency
- `avg_word_length`: Vocabulary complexity indicator
- **Normalization**: StandardScaler (z-score normalization)

### Classifier
- **Algorithm**: Logistic Regression with L2 regularization
- **Hyperparameter**: C = 3.0 (selected via dev set tuning)
- **Class Weighting**: Balanced (adjusts for class imbalance)
- **Solver**: SAGA (scales well for sparse features)

---

## Dataset Split

| Split | Size | Percentage | Fake % | Real % |
|-------|------|------------|--------|--------|
| **Train** | 352 | 69.8% | 47.4% | 52.6% |
| **Dev** | 76 | 15.1% | 47.4% | 52.6% |
| **Test** | 76 | 15.1% | 47.4% | 52.6% |

✅ **Stratified**: Class proportions maintained across splits  
✅ **Fixed seed**: Reproducible (RANDOM_SEED = 42)

---

## Performance Results

### Test Set Metrics

#### Overall Classification Performance
```
Accuracy:        86.84%
Macro-F1:        0.8683
Macro-Precision: 0.8684
Macro-Recall:    0.8694
```

#### Per-Class Performance

**Real News (Label = 0)**
- Precision: 0.8947 → Of articles predicted real, 89.5% are actually real
- Recall: 0.8500 → Of actual real articles, 85.0% are caught
- F1-Score: 0.8718
- Support: 40 articles

**Fake News (Label = 1)**
- Precision: 0.8421 → Of articles predicted fake, 84.2% are actually fake
- Recall: 0.8889 → Of actual fake articles, 88.9% are caught
- F1-Score: 0.8649
- Support: 36 articles

**Insight**: Slightly better at catching fake news (88.9% recall) but with more false alarms. Balanced overall.

#### Ranking Metrics (Treating as Information Retrieval)
```
ROC-AUC:  0.9257  → Excellent discrimination ability
PR-AUC:   0.8980  → Strong precision-recall tradeoff
```

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real     34      6     (85.0% correct)
       Fake      4     32     (88.9% correct)
```

**Analysis**:
- **True Positives (Fake caught)**: 32 / 36 = 88.9%
- **True Negatives (Real identified)**: 34 / 40 = 85.0%
- **False Positives (Real wrongly flagged)**: 6 / 40 = 15.0%
- **False Negatives (Fake slipped through)**: 4 / 36 = 11.1%

---

## Hyperparameter Tuning Results

Tested C values on development set:

| C Value | Dev Accuracy | Dev Macro-F1 | Dev PR-AUC | Selected? |
|---------|--------------|--------------|------------|-----------|
| 0.1 | 0.8158 | 0.8157 | 0.7650 | ❌ |
| 0.3 | 0.8026 | 0.8026 | 0.8144 | ❌ |
| 1.0 | 0.8553 | 0.8552 | 0.8676 | ❌ |
| **3.0** | **0.8684** | **0.8683** | **0.8692** | ✅ **Best F1** |
| 10.0 | 0.8553 | 0.8552 | 0.8912 | ❌ |

**Selection Criterion**: Maximum Macro-F1 on dev set → C = 3.0

---

## Error Analysis

### Error Distribution
- **False Positives** (Real predicted as Fake): 6 articles
- **False Negatives** (Fake predicted as Real): 4 articles
- **Total Errors**: 10 / 76 = 13.16%

### Most Confident Errors

#### Top 5 False Positives (Real News Wrongly Flagged as Fake)
1. **[97.7%]** "VIDEO: Special Preview Of Jersey Shore In Miami!"
   - *Analysis*: Clickbait-style title, entertainment content
   
2. **[69.0%]** "Leonard Lance claims federal tax code contains 4 million words..."
   - *Analysis*: Fact-checking article structure may resemble fake news patterns
   
3. **[65.4%]** "Strong Words in Ohio as Obama and Clinton Press On"
   - *Analysis*: Political language, shorter article
   
4. **[64.5%]** "Full text: Jeff Flake on Trump speech transcript"
   - *Analysis*: Political content, transcript format
   
5. **[54.3%]** "To young people who are undocumented: This is your country, too."
   - *Analysis*: Opinion piece, emotional language

**Pattern**: Real articles with sensational titles or political content get misclassified.

#### Top 4 False Negatives (Fake News Missed)
1. **[2.4%]** "The hacked emails at the center of Mueller's Russia investigation..."
   - *Analysis*: Professional writing style, long-form content
   
2. **[25.4%]** (Empty/minimal title)
   - *Analysis*: Missing metadata confuses classifier
   
3. **[28.5%]** "10 Things You Didn't Know About Nancy Pelosi"
   - *Analysis*: Listicle format, looks legitimate
   
4. **[43.9%]** "Ravi for Hoboken"
   - *Analysis*: Short, ambiguous title

**Pattern**: Fake articles that mimic professional journalism slip through.

---

## Feature Importance Analysis

### Top Fake News Indicators (Positive Coefficients)
Terms/features strongly associated with fake news:
- Political figures: "trump", "clinton", "obama"
- Sensational terms: "breaking", "shocking", "revealed"
- Stylometric: High uppercase ratio, short text length
- News sources: Certain domains/authors

### Top Real News Indicators (Negative Coefficients)
Terms/features strongly associated with real news:
- Professional language: "according", "statement", "official"
- Longer text (high word count)
- Lower punctuation ratio
- Fact-checking terms: "fact", "claim", "rating"

**Insight**: Model learned both content (words) and style (length, formatting) patterns effectively.

---

## Comparison to Baselines

| Model | Accuracy | Performance |
|-------|----------|-------------|
| **Random** | 50.0% | Coin flip |
| **Majority Class** | 52.6% | Always predict "real" |
| **Our Model** | **86.84%** | **+34.2% improvement** |

**Statistical Significance**: Model significantly outperforms both baselines (p < 0.001 by binomial test).

---

## Course Alignment (CS410 Topics)

### ✅ Implemented Topics

| Topic | Application | Location |
|-------|-------------|----------|
| **L3: VSM & TF-IDF** | Core feature representation | `train_baseline.py` |
| **L3: Document Normalization** | L2 norm on TF-IDF vectors | TfidfVectorizer |
| **L3: Term Weighting** | Sublinear TF, IDF weighting | TF-IDF setup |
| **L10: Classification Metrics** | Accuracy, P, R, F1 | `evaluate.py` |
| **L10: Ranking Metrics** | ROC-AUC, PR-AUC, AP | `evaluate.py` |
| **L10: Evaluation** | Train/dev/test split, curves | Both scripts |

### 🚀 Future Enhancements (Planned)

| Topic | Enhancement | Status |
|-------|-------------|--------|
| **L4-L5: Word2Vec** | Add pre-trained embeddings | Planned |
| **L6-L7: Language Models** | Unigram LM scoring | Planned |
| **L8: Feedback** | Rocchio feature reweighting | Planned |
| **Social Context** | User, propagation features | Future |

---

## Visualizations Generated

### 1. Confusion Matrix (`confusion_matrix.png`)
- Shows classification breakdown
- Annotated with counts and percentages

### 2. Precision-Recall Curve (`pr_curve.png`)
- Trade-off between precision and recall at different thresholds
- AP (Average Precision) = 0.8980

### 3. ROC Curve (`roc_curve.png`)
- True Positive Rate vs. False Positive Rate
- AUC = 0.9257 (excellent discrimination)

### 4. Feature Importance (`feature_importance.png`)
- Top 20 features for fake vs. real news
- Shows both content and style features

### 5. Probability Distribution (`probability_distribution.png`)
- Distribution of predicted probabilities for each class
- Shows model confidence patterns

---

## Strengths

✅ **High Accuracy**: 86.84% on test set  
✅ **Balanced Performance**: Good F1 for both classes  
✅ **Excellent Discrimination**: ROC-AUC = 0.93  
✅ **Interpretable**: Logistic regression coefficients show reasoning  
✅ **Fast**: Training takes seconds on CPU  
✅ **Reproducible**: Fixed seeds, saved models  
✅ **Course-Aligned**: Directly applies CS410 concepts  

---

## Limitations

❌ **Entertainment Content**: Struggles with non-news articles (Jersey Shore)  
❌ **Sophisticated Fakes**: Misses well-written fake articles  
❌ **Small Dataset**: Only 504 articles after cleaning  
❌ **No Social Context**: Missing user engagement, propagation data  
❌ **Static Features**: No temporal, multimodal (image) information  
❌ **English Only**: No multilingual support  

---

## Recommendations

### For Immediate Improvement
1. **Collect More Data**: Current 504 → Target 5,000+ articles
2. **Clean Entertainment**: Filter non-news content or create separate category
3. **Add Social Features**: When available from FakeNewsNet
4. **Ensemble Methods**: Combine with LM-based scoring (future work)

### For Production Deployment
1. **Calibration**: Ensure probabilities reflect true confidence
2. **Threshold Tuning**: Adjust decision boundary for precision/recall needs
3. **A/B Testing**: Compare with existing systems
4. **Monitoring**: Track performance drift over time
5. **Human-in-Loop**: Flag uncertain cases (0.4-0.6 probability) for review

### For Research Extension
1. **Word Embeddings**: Test Word2Vec, GloVe, FastText (L4-L5)
2. **Language Models**: Implement unigram LM approach (L6-L7)
3. **Feedback Mechanisms**: Apply Rocchio on train centroids (L8)
4. **Deep Learning**: Fine-tune BERT/RoBERTa for comparison
5. **Multi-Modal**: Integrate image analysis for memes, fake photos

---

## Files Generated

### Models & Vectorizers
```
outputs/
├── models/
│   └── baseline_model.pkl           # Trained Logistic Regression
└── vectorizers/
    ├── tfidf_body.pkl               # Body text vectorizer
    ├── tfidf_title.pkl              # Title vectorizer
    └── style_scaler.pkl             # Feature scaler
```

### Results & Analysis
```
outputs/
├── results/
│   ├── baseline_results.json        # All metrics
│   ├── error_analysis.json          # Detailed errors
│   └── test_predictions.csv         # Full predictions
└── figures/
    ├── confusion_matrix.png         # Classification matrix
    ├── pr_curve.png                 # Precision-Recall curve
    ├── roc_curve.png                # ROC curve
    ├── feature_importance.png       # Top features
    └── probability_distribution.png # Confidence distribution
```

### Data Splits
```
data/
├── politifact_extracted.csv         # Original data
├── train.csv                        # Training set (352)
├── dev.csv                          # Development set (76)
└── test.csv                         # Test set (76)
```

---

## Conclusion

**Success Criteria Met**: ✅ All targets exceeded

The baseline model demonstrates that:
1. **Text content alone** is highly informative for fake news detection
2. **Simple features** (TF-IDF + style) achieve strong performance
3. **Classical ML** (Logistic Regression) remains competitive
4. **CS410 techniques** (VSM, TF-IDF, evaluation) are directly applicable

**Next Steps**: Implement Word2Vec and Language Model enhancements per MODEL_PLAN.md Phase 6.

---

## Reproducibility

To reproduce these results:
```bash
cd UIUC-CS410-fake_news_detection/scripts
py train_baseline.py  # Training
py evaluate.py         # Evaluation & visualization
```

All random seeds fixed (RANDOM_SEED = 42) for reproducibility.

---

**Report Generated**: October 22, 2024  
**Model Version**: v0 (Baseline)  
**Status**: ✅ Complete and Validated

