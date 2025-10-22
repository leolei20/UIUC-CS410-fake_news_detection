# UIUC CS410 - Fake News Detection

[![Status](https://img.shields.io/badge/Status-Model%20Implemented-success)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-86.84%25-brightgreen)]()
[![F1](https://img.shields.io/badge/F1--Score-0.87-blue)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.93-orange)]()

This repository contains a complete fake news detection system using data from the [FakeNewsNet dataset](https://github.com/KaiDMML/FakeNewsNet) (PolitiFact subset). The project implements text retrieval and classification techniques learned in CS410 to distinguish fake news from real news articles.

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the trained model
cd scripts
py train_baseline.py   # Train the model (already done)
py evaluate.py          # Generate visualizations

# Or explore the data
py data_analysis.py     # Exploratory data analysis
```

## 📁 Project Structure

```
UIUC-CS410-fake_news_detection/
│
├── data/                              # Dataset and splits
│   ├── politifact_extracted.csv       # Original extracted articles (504)
│   ├── train.csv                      # Training set (352 articles, 70%)
│   ├── dev.csv                        # Development set (76 articles, 15%)
│   └── test.csv                       # Test set (76 articles, 15%)
│
├── scripts/                           # Python scripts
│   ├── preprocessing.py               # Original data extraction
│   ├── data_analysis.py               # Exploratory data analysis
│   ├── train_baseline.py              # Model training pipeline ⭐
│   └── evaluate.py                    # Model evaluation & visualization ⭐
│
├── outputs/                           # Generated results
│   ├── models/
│   │   └── baseline_model.pkl         # Trained Logistic Regression model
│   ├── vectorizers/
│   │   ├── tfidf_body.pkl             # Body text TF-IDF (5000 features)
│   │   ├── tfidf_title.pkl            # Title TF-IDF (500 features)
│   │   └── style_scaler.pkl           # Style feature scaler
│   ├── results/
│   │   ├── baseline_results.json      # All evaluation metrics
│   │   ├── error_analysis.json        # Detailed error analysis
│   │   └── test_predictions.csv       # Full predictions on test set
│   └── figures/
│       ├── confusion_matrix.png       # Classification performance
│       ├── pr_curve.png               # Precision-Recall curve
│       ├── roc_curve.png              # ROC curve
│       ├── feature_importance.png     # Top predictive features
│       └── probability_distribution.png
│
├── MODEL_PLAN.md                      # Detailed methodology & pipeline
├── RESULTS.md                         # Comprehensive results report
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── .gitignore                         # Git ignore rules
```

## 🚀 Model Performance

### Baseline Model (v0): TF-IDF + Logistic Regression

**Test Set Results** (76 articles):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | **86.84%** | Correctly classifies 66/76 articles |
| **Macro-F1** | **0.8683** | Balanced performance across classes |
| **ROC-AUC** | **0.9257** | Excellent discrimination ability |
| **PR-AUC** | **0.8980** | Strong precision-recall trade-off |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Real** | 0.8947 | 0.8500 | 0.8718 | 40 |
| **Fake** | 0.8421 | 0.8889 | 0.8649 | 36 |

**Confusion Matrix:**
```
                Predicted
              Real    Fake
Actual Real     34      6    (85.0% correct)
       Fake      4     32    (88.9% correct)
```

### Key Achievements
✅ **34% improvement** over majority baseline (52.6% → 86.8%)  
✅ **All targets exceeded** (Accuracy > 80%, F1 > 0.75, PR-AUC > 0.85)  
✅ **Balanced performance** across fake and real news classes  
✅ **Interpretable features** showing why articles are classified  

## 📊 Dataset Overview

- **Total Articles**: 504 (after cleaning)
- **Fake News**: 239 (47.4%)
- **Real News**: 265 (52.6%)
- **Source**: PolitiFact via [FakeNewsNet repository](https://github.com/KaiDMML/FakeNewsNet)

### Data Fields
- `id`: Article identifier
- `title`: Article title/headline
- `text`: Full article body text
- `author`: Article author(s) (56% missing)
- `label`: Ground truth classification (fake/real)
- `clean_text`: Preprocessed text (stopwords removed, lowercase)

### Key Characteristics from EDA

| Metric | Fake News | Real News | Insight |
|--------|-----------|-----------|---------|
| Avg Text Length | 2,339 chars | 14,429 chars | Real news more detailed |
| Avg Word Count | ~350 words | ~2,100 words | 6x longer |
| Avg Title Length | 11.6 words | 7.5 words | Fake titles more clickbait |
| Top Words | trump, president, clinton | people, going, think | Different focus |

## 🔧 Model Architecture

### Features (Total: 5,507 dimensions)

**1. TF-IDF Text Features (5,500)**
- **Body Text**: 5,000 features (unigrams + bigrams)
  - Sublinear TF weighting: `tf = 1 + log(count)`
  - IDF weighting: `idf = log(N / df)`
  - L2 document normalization
  - Min document frequency: 3
  
- **Title**: 500 features (unigrams + bigrams)
  - Separate vectorizer to capture headline patterns
  - Min document frequency: 2

**2. Stylometric Features (7)**
- `text_length`: Total character count
- `word_count`: Total word count
- `title_word_count`: Title length
- `punctuation_ratio`: Exclamation/question mark frequency
- `uppercase_ratio`: ALL CAPS usage
- `digit_ratio`: Numeric character frequency
- `avg_word_length`: Vocabulary complexity

### Classifier
- **Algorithm**: Logistic Regression with L2 regularization
- **Hyperparameter**: C = 3.0 (tuned on dev set)
- **Class Weighting**: Balanced (handles imbalance)
- **Solver**: SAGA (efficient for sparse features)

**Mathematical Formulation:**
```
P(fake | article) = σ(w^T · x + b)

where:
  x = [TF-IDF_body | TF-IDF_title | style_features]
  w = learned weights
  σ = sigmoid function
```

## 💻 Usage Guide

### Prerequisites
```bash
pip install -r requirements.txt
```

**Dependencies:**
- pandas, numpy (data processing)
- scikit-learn (machine learning)
- matplotlib, seaborn (visualization)
- nltk, wordcloud (text analysis)

### Training the Model

```bash
cd scripts
py train_baseline.py
```

**What it does:**
1. Loads and splits data (70/15/15 train/dev/test)
2. Extracts TF-IDF features (5,500 dimensions)
3. Extracts stylometric features (7 dimensions)
4. Tunes hyperparameter C on dev set
5. Trains final model with best C
6. Evaluates on test set
7. Saves model and vectorizers

**Output:**
- `outputs/models/baseline_model.pkl`
- `outputs/vectorizers/*.pkl`
- `outputs/results/baseline_results.json`

### Evaluating the Model

```bash
cd scripts
py evaluate.py
```

**What it does:**
1. Loads trained model and test data
2. Generates predictions and probabilities
3. Creates 5 visualizations
4. Performs error analysis
5. Identifies top confident errors

**Output:**
- 5 PNG visualizations in `outputs/figures/`
- Error analysis in `outputs/results/`

### Exploratory Data Analysis

```bash
cd scripts
py data_analysis.py
```

Generates comprehensive EDA visualizations and statistics.

## 📈 Key Findings

### What the Model Learned

**Top Fake News Indicators:**
- Political figures: "trump", "clinton", "obama"
- Sensational terms: "breaking", "shocking"
- Shorter text length
- Higher uppercase and punctuation ratios
- Clickbait-style titles

**Top Real News Indicators:**
- Professional language: "according", "statement"
- Longer, more detailed articles
- Lower punctuation ratio
- Fact-checking terms: "fact", "claim", "rating"
- Concise headlines

### Error Analysis

**False Positives (Real flagged as Fake): 6 articles**
- Entertainment content (e.g., "Jersey Shore" video)
- Opinion pieces with emotional language
- Political transcripts with clickbait-style titles

**False Negatives (Fake missed): 4 articles**
- Professional writing style mimicking journalism
- Longer articles with detailed content
- Missing or ambiguous metadata

## 🎓 Course Alignment (CS410)

This project directly applies concepts from CS 410 - Text Information Systems:

| Lecture | Concept | Application in Project |
|---------|---------|------------------------|
| **L1-L2** | Text Retrieval, Ranking | Treat classification as ranking by probability |
| **L3** | VSM, TF-IDF | Core feature representation (5,500 features) |
| **L3** | Document Normalization | L2 normalization of TF-IDF vectors |
| **L3** | Term Weighting | Sublinear TF, IDF calculation |
| **L10** | Evaluation Metrics | Accuracy, P, R, F1, ROC-AUC, PR-AUC |
| **L10** | PR Curves | Precision-Recall trade-off analysis |

### Future Enhancements (Planned)

| Lecture | Enhancement | Status |
|---------|-------------|--------|
| **L4-L5** | Word2Vec Embeddings | 📋 Planned |
| **L6-L7** | Language Models | 📋 Planned |
| **L8** | Rocchio Feedback | 📋 Planned |
| **Advanced** | Social Context Features | 🔮 Future |

See `MODEL_PLAN.md` for detailed methodology and future roadmap.

## 📊 Visualizations

All visualizations are automatically generated by `evaluate.py`:

1. **Confusion Matrix** - Classification performance breakdown
2. **Precision-Recall Curve** - Trade-off at different thresholds (AP = 0.898)
3. **ROC Curve** - True positive vs. false positive rate (AUC = 0.926)
4. **Feature Importance** - Top 20 features for fake/real classification
5. **Probability Distribution** - Model confidence patterns

View them in `outputs/figures/`

## 🔬 Reproducibility

All experiments use fixed random seeds for reproducibility:
- `RANDOM_SEED = 42` in all scripts
- Stratified splits maintain class balance
- Saved models and vectorizers ensure consistency

**To reproduce results:**
```bash
cd scripts
py train_baseline.py  # Will generate identical splits and model
py evaluate.py         # Will generate identical metrics
```

## 📚 Documentation

- **[MODEL_PLAN.md](MODEL_PLAN.md)** - Detailed methodology, mathematical formulations, pipeline
- **[RESULTS.md](RESULTS.md)** - Comprehensive results report with analysis
- **[requirements.txt](requirements.txt)** - All Python dependencies

## 🚀 Future Work

### Immediate Next Steps (Model v1)
1. **Word Embeddings** (L4-L5)
   - Add Word2Vec or GloVe features
   - Compare with TF-IDF performance
   
2. **Language Models** (L6-L7)
   - Build unigram LMs for fake/real classes
   - Score by log-likelihood ratio
   
3. **Feedback Mechanisms** (L8)
   - Apply Rocchio on train centroids
   - Reweight TF-IDF features

### Long-Term Enhancements
- **Social Context**: User credibility, tweet counts, propagation patterns
- **Deep Learning**: BERT, RoBERTa fine-tuning
- **Multimodal**: Image analysis for fake photos/memes
- **Temporal**: Track how fake news evolves over time
- **Explainability**: LIME/SHAP for individual predictions

## 🤝 Contributors

**CS 410 Final Project Team:**
- Fred Hou (xuzheh2@illinois.edu) - Feature Engineering
- Yixiang Zhang (yz100@illinois.edu) - Model Development
- Xiaoxi Chen (xiaoxic2@illinois.edu) - Data Preprocessing
- Xiaoxiong Lei (lei20@illinois.edu) - Evaluation & Testing

## 📄 References

1. Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2018). *FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media*. arXiv:1809.01286. [https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.

3. CS 410 Course Materials, UIUC - Text Information Systems

## 📝 License & Citation

This project is for educational purposes as part of CS 410 at UIUC.

If you use this code or methodology, please cite:
```bibtex
@misc{uiuc_cs410_fakenews_2024,
  title={Fake News Detection using TF-IDF and Logistic Regression},
  author={Hou, Fred and Zhang, Yixiang and Chen, Xiaoxi and Lei, Xiaoxiong},
  year={2024},
  institution={University of Illinois Urbana-Champaign},
  course={CS 410 - Text Information Systems}
}
```

## ⚙️ System Requirements

- **Python**: 3.8+
- **OS**: Windows 11 / macOS / Linux
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 500MB for data and models

---

**Course**: CS 410 - Text Information Systems  
**Institution**: University of Illinois Urbana-Champaign (UIUC)  
**Semester**: Fall 2024  
**Project Status**: ✅ Baseline Complete, Ready for Enhancements
