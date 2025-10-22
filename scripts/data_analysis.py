import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
print("Loading data...")
df = pd.read_csv('../data/politifact_extracted.csv')

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel Distribution:")
print(df['label'].value_counts())
print(f"\nPercentages:")
print(df['label'].value_counts(normalize=True) * 100)

# Basic statistics
print("\n" + "="*80)
print("TEXT LENGTH STATISTICS")
print("="*80)
df['text_length'] = df['text'].astype(str).str.len()
df['word_count'] = df['text'].astype(str).str.split().str.len()
df['clean_word_count'] = df['clean_text'].astype(str).str.split().str.len()

print("\nText Length by Label:")
print(df.groupby('label')[['text_length', 'word_count', 'clean_word_count']].describe())

# Missing values
print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(df.isnull().sum())

# Title analysis
print("\n" + "="*80)
print("TITLE ANALYSIS")
print("="*80)
df['title_length'] = df['title'].astype(str).str.len()
df['title_word_count'] = df['title'].astype(str).str.split().str.len()
print("\nAverage title length by label:")
print(df.groupby('label')['title_length'].mean())
print("\nAverage title word count by label:")
print(df.groupby('label')['title_word_count'].mean())

# Author analysis
print("\n" + "="*80)
print("AUTHOR ANALYSIS")
print("="*80)
author_counts = df['author'].value_counts()
print(f"Total unique authors: {df['author'].nunique()}")
print(f"Articles with no author: {df['author'].isna().sum()}")
print(f"\nTop 10 authors:")
print(author_counts.head(10))

# Word frequency analysis
print("\n" + "="*80)
print("WORD FREQUENCY ANALYSIS")
print("="*80)

def get_top_words(text_series, n=20):
    """Get top n most common words from a series of texts"""
    all_words = ' '.join(text_series.astype(str)).split()
    return Counter(all_words).most_common(n)

fake_news = df[df['label'] == 'fake']
real_news = df[df['label'] == 'real']

print("\nTop 20 words in FAKE news:")
fake_top_words = get_top_words(fake_news['clean_text'], 20)
for word, count in fake_top_words:
    print(f"  {word}: {count}")

print("\nTop 20 words in REAL news:")
real_top_words = get_top_words(real_news['clean_text'], 20)
for word, count in real_top_words:
    print(f"  {word}: {count}")

# Visualization 1: Label Distribution
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Label distribution
ax1 = axes[0, 0]
label_counts = df['label'].value_counts()
ax1.bar(label_counts.index, label_counts.values, color=['#ff6b6b', '#4ecdc4'])
ax1.set_xlabel('Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Fake vs Real News', fontsize=14, fontweight='bold')
for i, v in enumerate(label_counts.values):
    ax1.text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 2: Text length distribution
ax2 = axes[0, 1]
ax2.hist([fake_news['text_length'], real_news['text_length']], 
         label=['Fake', 'Real'], bins=30, alpha=0.7, color=['#ff6b6b', '#4ecdc4'])
ax2.set_xlabel('Text Length (characters)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Text Length Distribution', fontsize=14, fontweight='bold')
ax2.legend()

# Plot 3: Word count distribution
ax3 = axes[1, 0]
ax3.boxplot([fake_news['word_count'], real_news['word_count']], 
            tick_labels=['Fake', 'Real'], patch_artist=True,
            boxprops=dict(facecolor='#95e1d3'))
ax3.set_ylabel('Word Count', fontsize=12, fontweight='bold')
ax3.set_title('Word Count Comparison', fontsize=14, fontweight='bold')

# Plot 4: Title length comparison
ax4 = axes[1, 1]
ax4.violinplot([fake_news['title_word_count'].dropna(), 
                real_news['title_word_count'].dropna()], 
               showmeans=True, showmedians=True)
ax4.set_xticks([1, 2])
ax4.set_xticklabels(['Fake', 'Real'])
ax4.set_ylabel('Title Word Count', fontsize=12, fontweight='bold')
ax4.set_title('Title Length Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/data_analysis_overview.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: outputs/data_analysis_overview.png")

# Visualization 2: Word clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Fake news word cloud
fake_text = ' '.join(fake_news['clean_text'].astype(str))
wordcloud_fake = WordCloud(width=800, height=400, background_color='white', 
                           colormap='Reds', max_words=100).generate(fake_text)
axes[0].imshow(wordcloud_fake, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('FAKE NEWS - Most Common Words', fontsize=16, fontweight='bold')

# Real news word cloud
real_text = ' '.join(real_news['clean_text'].astype(str))
wordcloud_real = WordCloud(width=800, height=400, background_color='white',
                           colormap='Blues', max_words=100).generate(real_text)
axes[1].imshow(wordcloud_real, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('REAL NEWS - Most Common Words', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/wordclouds.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: outputs/wordclouds.png")

# Visualization 3: Top words comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top words in fake news
fake_words_df = pd.DataFrame(fake_top_words, columns=['word', 'count'])
axes[0].barh(fake_words_df['word'][:15], fake_words_df['count'][:15], color='#ff6b6b')
axes[0].set_xlabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Top 15 Words in FAKE News', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Top words in real news
real_words_df = pd.DataFrame(real_top_words, columns=['word', 'count'])
axes[1].barh(real_words_df['word'][:15], real_words_df['count'][:15], color='#4ecdc4')
axes[1].set_xlabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Top 15 Words in REAL News', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('../outputs/top_words_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: outputs/top_words_comparison.png")

# Export detailed statistics to CSV
print("\n" + "="*80)
print("EXPORTING DETAILED STATISTICS")
print("="*80)

stats_summary = pd.DataFrame({
    'Metric': ['Total Articles', 'Fake News', 'Real News', 'Avg Text Length (Fake)', 
               'Avg Text Length (Real)', 'Avg Word Count (Fake)', 'Avg Word Count (Real)',
               'Unique Authors'],
    'Value': [len(df), len(fake_news), len(real_news), 
              fake_news['text_length'].mean(), real_news['text_length'].mean(),
              fake_news['word_count'].mean(), real_news['word_count'].mean(),
              df['author'].nunique()]
})
stats_summary.to_csv('../outputs/statistics_summary.csv', index=False)
print("[OK] Saved: outputs/statistics_summary.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files in outputs/ folder:")
print("  1. data_analysis_overview.png - Overview visualizations")
print("  2. wordclouds.png - Word clouds for fake and real news")
print("  3. top_words_comparison.png - Top words comparison")
print("  4. statistics_summary.csv - Detailed statistics")

