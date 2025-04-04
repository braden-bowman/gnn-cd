#\!/usr/bin/env python3
# Comprehensive evaluation of UNSW-NB15 results

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from pathlib import Path

# Create results directory
results_dir = "data/unsw/results/evaluation"
os.makedirs(results_dir, exist_ok=True)

# Load the trained models
print("Loading models from data/unsw/results/community_detection_results.pkl")
with open('data/unsw/results/community_detection_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Function to create comparison dataframe
def create_comparison_df(results):
    """Create a DataFrame with model comparison metrics"""
    rows = []
    for model_name, result in results.items():
        metrics = result['metrics']
        row = {
            'Method': model_name,
            'Num Communities': metrics['num_communities'],
            'Avg Community Size': metrics['avg_community_size'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'], 
            'F1': metrics['f1'],
            'AUC': metrics.get('auc', 0),
            'Avg Purity': metrics['avg_purity'],
            'Attack Comm. Ratio': metrics.get('attack_communities_ratio', 0),
            'Execution Time (s)': result['execution_time']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

# Create comparison dataframe and save it
comparison_df = create_comparison_df(results)
print("\nModel Comparison:")
print(comparison_df)
comparison_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

# Create visualizations
print("\nCreating visualizations...")

# Performance metrics
plt.figure(figsize=(14, 8))
metrics_df = comparison_df.melt(
    id_vars=['Method'],
    value_vars=['Accuracy', 'Precision', 'Recall', 'F1', 'Avg Purity'],
    var_name='Metric',
    value_name='Value'
)
sns.barplot(x='Method', y='Value', hue='Metric', data=metrics_df)
plt.title('Performance Metrics by Method')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'performance_metrics.png'))
plt.close()

# Community structure metrics
plt.figure(figsize=(14, 6))
structure_df = comparison_df.melt(
    id_vars=['Method'],
    value_vars=['Num Communities', 'Avg Community Size', 'Attack Comm. Ratio'],
    var_name='Metric',
    value_name='Value'
)
g = sns.catplot(x='Method', y='Value', col='Metric', data=structure_df, kind='bar',
               sharey=False, height=5, aspect=0.8)
g.set_xticklabels(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'community_structure.png'))
plt.close()

# Execution time comparison
plt.figure(figsize=(12, 6))
time_plot = sns.barplot(x='Method', y='Execution Time (s)', data=comparison_df)
plt.title('Execution Time Comparison')
plt.yscale('log')
plt.xticks(rotation=45)
for i, bar in enumerate(time_plot.patches):
    time_plot.text(
        bar.get_x() + bar.get_width()/2., 
        bar.get_height() + 0.01, 
        f"{bar.get_height():.4f}s", 
        ha='center', va='bottom', rotation=0
    )
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'execution_time.png'))
plt.close()

# F1 Score vs Execution Time scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Execution Time (s)', y='F1', 
               size='Num Communities', hue='Method', 
               data=comparison_df, sizes=(50, 500), alpha=0.8)
plt.xscale('log')
plt.title('F1 Score vs Execution Time')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'f1_vs_time.png'))
plt.close()

# Heatmap of methods vs metrics
metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Avg Purity']
plt.figure(figsize=(12, 8))
heatmap_df = comparison_df.set_index('Method')[metrics_cols]
sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.3f', linewidths=0.5)
plt.title('Performance Comparison of Community Detection Methods')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'methods_heatmap.png'))
plt.close()

# Comprehensive ranking by metrics
def rank_methods(df, metrics):
    """Rank methods by multiple metrics"""
    ranks = pd.DataFrame(index=df['Method'])
    
    # For each metric, calculate ranks (higher is better)
    for metric in metrics:
        ranks[f"{metric}_rank"] = df[metric].rank(ascending=True if metric == 'Execution Time (s)' else False)
    
    # Calculate average rank
    rank_cols = [col for col in ranks.columns if col.endswith('_rank')]
    ranks['avg_rank'] = ranks[rank_cols].mean(axis=1)
    
    # Sort by average rank
    return ranks.sort_values('avg_rank')

# Create ranking
print("\nRanking methods by performance...")
ranking_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Avg Purity', 'Execution Time (s)']
rankings = rank_methods(comparison_df, ranking_metrics)
print(rankings)

# Save ranking to file
rankings.to_csv(os.path.join(results_dir, "method_rankings.csv"))

print(f"\nAll visualizations and analyses saved to {results_dir}")
