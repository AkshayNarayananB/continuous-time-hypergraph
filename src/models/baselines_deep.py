import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from layers.temporal_memory import NodeMemoryUpdater # From our previous step

class HypergraphSpectralClustering:
    """
    Step 7: Static Hypergraph Spectral Clustering Baseline.
    Builds the normalized hypergraph Laplacian and clusters its eigenvectors.
    """
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def fit_predict(self, num_nodes: int, hyperedges: list) -> np.ndarray:
        # 1. Build incidence matrix H
        row_idx, col_idx = [], []
        for e_idx, edge_nodes in enumerate(hyperedges):
            for node in edge_nodes:
                row_idx.append(node)
                col_idx.append(e_idx)
                
        H = sp.csr_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(num_nodes, len(hyperedges)))
        
        # 2. Compute Degree Matrices
        D_v_arr = np.array(H.sum(axis=1)).flatten()
        D_e_arr = np.array(H.sum(axis=0)).flatten()
        
        # Add epsilon to prevent division by zero
        D_v_inv_sqrt = sp.diags(1.0 / np.sqrt(D_v_arr + 1e-8))
        D_e_inv = sp.diags(1.0 / (D_e_arr + 1e-8))
        
        # 3. Normalized Hypergraph Laplacian: L = I - D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        Theta = D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt
        I = sp.eye(num_nodes)
        L = I - Theta
        
        # 4. Extract Spectral Embeddings (Eigenvectors of K smallest eigenvalues)
        # We use 'SM' (Smallest Magnitude) and extract K+1, dropping the trivial first eigenvector
        eigenvalues, eigenvectors = eigsh(L, k=self.n_clusters + 1, which='SM')
        embeddings = eigenvectors[:, 1:] 
        
        # 5. Cluster the embeddings
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

class TGNCliqueBaseline(nn.Module):
    """
    Step 8: Temporal Graph Network (TGN) Baseline.
    Demonstrates the flaw of breaking hyperedges into pairwise cliques for continuous memory.
    """
    def __init__(self, num_nodes: int, memory_dim: int, time_dim: int):
        super().__init__()
        # Reusing the memory module, but feeding it pairwise edges instead of hyperedges
        self.memory_updater = NodeMemoryUpdater(num_nodes, memory_dim, time_dim)
        self.node_embedder = nn.Linear(memory_dim, memory_dim)

    def forward(self, hyperedge_nodes: list, current_t: float):
        # FLAW IDENTIFICATION: Explode the hyperedge into pairwise cliques (O(N^2) memory waste)
        pairwise_edges = []
        for i in range(len(hyperedge_nodes)):
            for j in range(i + 1, len(hyperedge_nodes)):
                pairwise_edges.append((hyperedge_nodes[i], hyperedge_nodes[j]))
        
        # Create dummy message (e.g., all ones) for each node interaction
        t_tensor = torch.tensor([current_t] * len(pairwise_edges), dtype=torch.float32)
        
        # Update memory for each pair (simplified for baseline purposes)
        # In a full TGN, this iterates through all pairs, corrupting group semantics
        flattened_nodes = torch.tensor([n for pair in pairwise_edges for n in pair], dtype=torch.long)
        messages = torch.ones((len(flattened_nodes), self.memory_updater.memory_dim))
        t_expanded = torch.repeat_interleave(t_tensor, 2)
        
        self.memory_updater(flattened_nodes, messages, t_expanded)
        
    def get_embeddings(self):
        return self.node_embedder(self.memory_updater.get_memory()).detach().cpu().numpy()