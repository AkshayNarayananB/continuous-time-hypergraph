import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from networkx.algorithms.community import louvain_communities
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from utils import evaluate_clustering
import warnings

warnings.filterwarnings("ignore")

# Setup plot directory
os.makedirs("../plots", exist_ok=True)

# ---------------------------------------------------------
# PHASE 1: LOAD DATA & EDA
# ---------------------------------------------------------
print("--- Loading Synthetic Data ---")
df_edges = pd.read_pickle("../data/synthetic/synthetic_edges.pkl")
df_labels = pd.read_pickle("../data/synthetic/synthetic_labels.pkl")

# We evaluate on the final time step to see how models handle the accumulated graph
final_t = df_labels['time_step'].max()
ground_truth = df_labels[df_labels['time_step'] == final_t].sort_values('node_id')['cluster'].values
num_nodes = len(ground_truth)

print("\n--- Running Exploratory Data Analysis (EDA) ---")
# 1. Cardinality Distribution (Higher-Order Proof)
df_edges['k_size'] = df_edges['hyperedge_nodes'].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df_edges['k_size'], bins=range(2, df_edges['k_size'].max() + 2), discrete=True)
plt.title("Hyperedge Cardinality Distribution\n(Proves interactions are not just pairwise)")
plt.xlabel("Number of Nodes in Hyperedge (k)")
plt.ylabel("Frequency")
plt.savefig("../plots/eda_cardinality.png")
plt.close()

# 2. Temporal Density (Non-Stationary Proof)
temporal_density = df_edges.groupby('time_step').size()
plt.figure(figsize=(10, 4))
plt.plot(temporal_density.index, temporal_density.values, marker='o', linestyle='-', color='firebrick')
plt.axvline(x=final_t//2, color='k', linestyle='--', label='Injected Concept Drift')
plt.title("Hyperedge Arrival Rate over Time\n(Shows streaming density & drift points)")
plt.xlabel("Time Step")
plt.ylabel("Number of Hyperedges")
plt.legend()
plt.savefig("../plots/eda_temporal_density.png")
plt.close()

print("EDA visualizations saved to '../plots/'")

# ---------------------------------------------------------
# PHASE 2: STATIC GRAPH PROJECTION & FEATURE EXTRACTION
# ---------------------------------------------------------
print("\n--- Extracting Features for Classical Baselines ---")
# To run classical baselines, we must project the hypergraph into a static pairwise graph
G = nx.Graph()
G.add_nodes_from(range(num_nodes))

for _, row in df_edges.iterrows():
    nodes = row['hyperedge_nodes']
    # Clique expansion: connect all nodes in the hyperedge
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if G.has_edge(nodes[i], nodes[j]):
                G[nodes[i]][nodes[j]]['weight'] += 1
            else:
                G.add_edge(nodes[i], nodes[j], weight=1)

# Extract Node Features for K-Means (Degree, Clustering Coefficient, Centrality)
print("Computing node topological features...")
degrees = dict(G.degree(weight='weight'))
clustering_coeffs = nx.clustering(G, weight='weight')
# Centrality is expensive; using degree centrality as proxy for speed
degree_centrality = nx.degree_centrality(G)

X_features = np.zeros((num_nodes, 3))
for i in range(num_nodes):
    X_features[i, 0] = degrees.get(i, 0)
    X_features[i, 1] = clustering_coeffs.get(i, 0)
    X_features[i, 2] = degree_centrality.get(i, 0)

# Normalize features
X_features = (X_features - X_features.mean(axis=0)) / (X_features.std(axis=0) + 1e-8)

# Compute Shortest Path Distance Matrix for DBSCAN
print("Computing shortest-path distance matrix...")
adj_matrix = nx.to_scipy_sparse_array(G, nodelist=range(num_nodes), weight='weight')
# Invert weights for distance (higher weight = shorter distance)
adj_matrix.data = 1.0 / adj_matrix.data
dist_matrix = shortest_path(csgraph=adj_matrix, directed=False, unweighted=False)
dist_matrix[np.isinf(dist_matrix)] = dist_matrix[~np.isinf(dist_matrix)].max() * 2 # Handle disconnected components

# ---------------------------------------------------------
# PHASE 3: EXECUTE BASELINES & EVALUATE
# ---------------------------------------------------------
print("\n--- Evaluating Baselines ---")

# 1. K-Means++
kmeans = KMeans(n_clusters=len(np.unique(ground_truth)), init='k-means++', n_init=10, random_state=42)
pred_kmeans = kmeans.fit_predict(X_features)
res_kmeans = evaluate_clustering(X_features, ground_truth, pred_kmeans, metric_type='euclidean')

# 2. DBSCAN (using precomputed shortest paths)
# Epsilon needs tuning based on the distance matrix scale
epsilon = np.percentile(dist_matrix, 10) 
dbscan = DBSCAN(eps=epsilon, min_samples=5, metric='precomputed')
pred_dbscan = dbscan.fit_predict(dist_matrix)
res_dbscan = evaluate_clustering(dist_matrix, ground_truth, pred_dbscan, metric_type='precomputed')

# 3. Louvain Community Detection (Graph-native baseline)
louvain_partition = louvain_communities(G, weight='weight')
pred_louvain = np.zeros(num_nodes, dtype=int)
for cluster_id, community in enumerate(louvain_partition):
    for node in community:
        pred_louvain[node] = cluster_id
res_louvain = evaluate_clustering(X_features, ground_truth, pred_louvain, metric_type='euclidean')

# ---------------------------------------------------------
# PHASE 4: RESULTS REPORTING
# ---------------------------------------------------------
print("\n================ FINAL BASELINE RESULTS ================")
print(f"{'Method':<15} | {'NMI':<6} | {'ARI':<6} | {'Silhouette':<10}")
print("-" * 45)
print(f"{'K-Means++':<15} | {res_kmeans['NMI']:.4f} | {res_kmeans['ARI']:.4f} | {res_kmeans['Silhouette']:.4f}")
print(f"{'DBSCAN':<15} | {res_dbscan['NMI']:.4f} | {res_dbscan['ARI']:.4f} | {res_dbscan['Silhouette']:.4f}")
print(f"{'Louvain':<15} | {res_louvain['NMI']:.4f} | {res_louvain['ARI']:.4f} | {res_louvain['Silhouette']:.4f}")
print("========================================================\n")