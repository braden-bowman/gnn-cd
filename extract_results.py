import pickle
import sys

with open('data/unsw/results/community_detection_results.pkl', 'rb') as f:
    results = pickle.load(f)

print('Models in results file:')
for model in results:
    print(f' - {model}')

print('\nSample metrics available:')
metrics = list(next(iter(results.values()))['metrics'].keys())
print(f' - {metrics}')

print('\nDetailed metrics by model:')
for model_name, model_data in results.items():
    metrics = model_data['metrics']
    communities = model_data['communities']
    size = len(communities) if isinstance(communities, list) else len(set(communities.values()))
    
    print(f"\n{model_name}:")
    print(f"  Communities: {size}")
    print(f"  Execution time: {model_data['execution_time']:.4f}s")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    if 'attack_communities_ratio' in metrics:
        print(f"  Attack communities ratio: {metrics['attack_communities_ratio']:.4f}")
