from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import numpy as np

def evaluate_clustering(X, true_labels, pred_labels, metric_type='euclidean'):
    """Computes internal and external clustering metrics."""
    results = {}
    
    # External Metrics (Requires ground truth)
    if true_labels is not None:
        results['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
        results['ARI'] = adjusted_rand_score(true_labels, pred_labels)
    else:
        results['NMI'], results['ARI'] = None, None

    # Internal Metric (Requires feature matrix or distance matrix)
    # If pred_labels has only 1 cluster, silhouette score is undefined
    if len(np.unique(pred_labels)) > 1:
        results['Silhouette'] = silhouette_score(X, pred_labels, metric=metric_type)
    else:
        results['Silhouette'] = -1.0
        
    return results