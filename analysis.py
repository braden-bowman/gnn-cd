#\!/usr/bin/env python3
# Analyze existing UNSW model results

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# Create results directory
results_dir = "data/unsw/results/analysis"
os.makedirs(results_dir, exist_ok=True)

# Load trained models
print("Loading trained models...")
with open('data/unsw/results/community_detection_results.pkl', 'rb') as f:
    models = pickle.load(f)

# Function to extract metrics from model results
def extract_metrics(models):
    """Extract metrics from model results into a DataFrame"""
    rows = []
    for model_name, model_data in models.items():
        metrics = model_data['metrics']
        row = {
            'Method': model_name,
            'Communities': metrics['num_communities'],
            'Avg Size': metrics['avg_community_size'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'AUC': metrics.get('auc', 0),
            'Purity': metrics['avg_purity'],
            'Attack Ratio': metrics.get('attack_communities_ratio', 0),
            'Execution Time': model_data['execution_time']
        }
        rows.append(row)
    return pd.DataFrame(rows)

# Extract metrics
metrics_df = extract_metrics(models)
print("\nModel performance metrics:")
print(metrics_df)
metrics_df.to_csv(os.path.join(results_dir, 'model_metrics.csv'), index=False)

# Create visualizations
print("\nCreating visualizations...")

# 1. Performance metrics comparison
plt.figure(figsize=(14, 8))
perf_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Purity']
perf_df = metrics_df.melt(id_vars=['Method'], value_vars=perf_metrics, var_name='Metric', value_name='Value')
sns.barplot(x='Method', y='Value', hue='Metric', data=perf_df)
plt.title('Performance Metrics Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'performance_metrics.png'))
plt.close()

# 2. Community structure comparison
plt.figure(figsize=(14, 6))
structure_metrics = ['Communities', 'Avg Size', 'Attack Ratio']
structure_df = metrics_df.melt(id_vars=['Method'], value_vars=structure_metrics, var_name='Metric', value_name='Value')
g = sns.catplot(x='Method', y='Value', col='Metric', data=structure_df, kind='bar',
               sharey=False, height=5, aspect=0.8)
g.set_xticklabels(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'community_structure.png'))
plt.close()

# 3. Execution time comparison
plt.figure(figsize=(12, 6))
time_plot = sns.barplot(x='Method', y='Execution Time', data=metrics_df)
plt.title('Execution Time Comparison')
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

# 4. F1 vs Execution Time scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Execution Time', y='F1', 
               size='Communities', hue='Method', 
               data=metrics_df, sizes=(50, 500), alpha=0.8)
plt.title('F1 Score vs Execution Time')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'f1_vs_time.png'))
plt.close()

# 5. Model ranking
def rank_models(df, metrics, higher_better=True):
    """Rank models by various metrics"""
    ranks = {}
    for metric in metrics:
        # Sort by metric (ascending=False for higher is better)
        sorted_df = df.sort_values(metric, ascending=not higher_better)
        # Assign ranks
        ranks[metric] = {row['Method']: i+1 for i, row in enumerate(sorted_df.to_dict('records'))}
    
    # Calculate average rank
    methods = df['Method'].tolist()
    avg_ranks = {}
    for method in methods:
        avg_ranks[method] = np.mean([ranks[metric][method] for metric in metrics])
    
    # Sort by average rank
    return {k: v for k, v in sorted(avg_ranks.items(), key=lambda item: item[1])}

# Rank by performance
perf_ranking = rank_models(metrics_df, ['Accuracy', 'Precision', 'Recall', 'F1', 'Purity'])
print("\nModel ranking by performance metrics (lower is better):")
for method, rank in perf_ranking.items():
    print(f"{method}: {rank:.2f}")

# Rank by efficiency
efficiency_ranking = rank_models(metrics_df, ['Execution Time'], higher_better=False)
print("\nModel ranking by efficiency (lower is better):")
for method, rank in efficiency_ranking.items():
    print(f"{method}: {rank:.2f}")

# 6. Compare traditional vs GNN vs overlapping methods
method_types = {
    'Traditional': ['louvain', 'infomap', 'label_propagation'],
    'GNN-based': ['gcn'],
    'Overlapping': ['bigclam']
}

type_metrics = []
for type_name, methods in method_types.items():
    metrics = metrics_df[metrics_df['Method'].isin(methods)]
    if not metrics.empty:
        row = {
            'Method Type': type_name,
            'Avg Accuracy': metrics['Accuracy'].mean(),
            'Avg Precision': metrics['Precision'].mean(),
            'Avg Recall': metrics['Recall'].mean(),
            'Avg F1': metrics['F1'].mean(),
            'Avg Communities': metrics['Communities'].mean(),
            'Avg Execution Time': metrics['Execution Time'].mean()
        }
        type_metrics.append(row)

type_df = pd.DataFrame(type_metrics)
print("\nComparison by method type:")
print(type_df)
type_df.to_csv(os.path.join(results_dir, 'method_type_comparison.csv'), index=False)

# 7. Method type comparison visualization
plt.figure(figsize=(14, 8))
type_perf_df = type_df.melt(id_vars=['Method Type'], 
                           value_vars=['Avg Accuracy', 'Avg Precision', 'Avg Recall', 'Avg F1'],
                           var_name='Metric', value_name='Value')
sns.barplot(x='Method Type', y='Value', hue='Metric', data=type_perf_df)
plt.title('Performance by Method Type')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'method_type_performance.png'))
plt.close()

# 8. Analysis of attack community detection
print("\nAnalysis of attack community detection:")
for method_name, model_data in models.items():
    metrics = model_data['metrics']
    print(f"\n{method_name}:")
    print(f"  Attack communities ratio: {metrics.get('attack_communities_ratio', 0):.4f}")
    print(f"  Precision (true positive rate): {metrics['precision']:.4f}")
    print(f"  Recall (attack detection rate): {metrics['recall']:.4f}")

# 9. Create heatmap of models vs metrics
plt.figure(figsize=(12, 8))
heatmap_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Purity', 'Attack Ratio']
heatmap_df = metrics_df.set_index('Method')[heatmap_metrics]
sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.3f', linewidths=0.5)
plt.title('Performance Comparison of Community Detection Methods')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'methods_heatmap.png'))
plt.close()

print(f"\nAnalysis completed. Results saved to {results_dir}")
