# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.temporal_memory import NodeMemoryUpdater
# from layers.hypergraph_conv import HypergraphConv

# class TempHyperDEC(nn.Module):
#     def __init__(self, num_nodes: int, memory_dim: int, time_dim: int, hidden_dim: int, n_clusters: int, alpha: float = 1.0):
#         super().__init__()
#         self.num_nodes = num_nodes
#         self.n_clusters = n_clusters
#         self.alpha = alpha
        
#         # THE FIX: Unique Structural Identities (ID Badges) for every node
#         self.node_emb = nn.Embedding(num_nodes, memory_dim)
        
#         self.memory_module = NodeMemoryUpdater(num_nodes=num_nodes, memory_dim=memory_dim, time_dim=time_dim)
#         self.hypergraph_conv = HypergraphConv(in_channels=memory_dim, out_channels=hidden_dim)
        
#         self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
#         nn.init.xavier_uniform_(self.cluster_centers)
#         self.register_buffer('prev_cluster_centers', torch.zeros_like(self.cluster_centers))

#     def forward(self, node_indices: torch.Tensor, hyperedge_indices: torch.Tensor, current_t: torch.Tensor):
#         # Pass the node's unique embedding into the memory instead of anonymous "ones"
#         messages = self.node_emb(node_indices)
#         self.memory_module(node_indices, messages, current_t)
        
#         current_memory = self.memory_module.get_memory()
        
#         # Combine temporal memory with base structural identity
#         h = current_memory + self.node_emb.weight
        
#         z = self.hypergraph_conv(h, node_indices, hyperedge_indices)
#         z = F.normalize(z, p=2, dim=1) 
        
#         dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
#         q = 1.0 / (1.0 + dist / self.alpha)
#         q = q ** ((self.alpha + 1.0) / 2.0)
#         q = (q.t() / torch.sum(q, dim=1)).t()
        
#         return z, q

#     def update_prev_centroids(self):
#         self.prev_cluster_centers.data.copy_(self.cluster_centers.detach().data)


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.temporal_memory import NodeMemoryUpdater
from layers.hypergraph_conv import HypergraphConv, HypergraphAttention

class TempHyperDEC(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int, time_dim: int, hidden_dim: int, n_clusters: int, alpha: float = 1.0, use_attention: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.use_attention = use_attention
        
        self.node_emb = nn.Embedding(num_nodes, memory_dim)
        self.memory_module = NodeMemoryUpdater(num_nodes=num_nodes, memory_dim=memory_dim, time_dim=time_dim)
        
        if self.use_attention:
            self.hypergraph_layer = HypergraphAttention(in_channels=memory_dim, out_channels=hidden_dim)
        else:
            self.hypergraph_layer = HypergraphConv(in_channels=memory_dim, out_channels=hidden_dim)
        
        # Keep the clustering parameters intact just in case
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_centers)
        
        # --- THE NEW LINK PREDICTION HEAD ---
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Fusion layer to combine neural structural score with heuristic frequency score
        self.feature_fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, node_indices: torch.Tensor, hyperedge_indices: torch.Tensor, current_t: torch.Tensor):
        messages = self.node_emb(node_indices)
        self.memory_module(node_indices, messages, current_t)
        
        current_memory = self.memory_module.get_memory()
        h = current_memory + self.node_emb.weight
        
        z = self.hypergraph_layer(h, node_indices, hyperedge_indices)
        z = F.normalize(z, p=2, dim=1) 
        
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        
        return z, q

    def predict_links(self, z: torch.Tensor, node_indices: torch.Tensor, edge_indices: torch.Tensor, num_edges: int, heuristic_scores = None):
        """
        Pools the node embeddings for a given hyperedge and scores the likelihood of it existing.
        """
        gathered_z = z[node_indices]
        idx_edge = edge_indices.unsqueeze(1).expand_as(gathered_z)
        edge_reps = torch.zeros(num_edges, z.size(1), device=z.device)
        edge_reps.scatter_reduce_(0, idx_edge, gathered_z, reduce='mean', include_self=False)
        
        # 1. Get the neural network's structural logit score
        neural_scores = self.link_predictor(edge_reps).squeeze(-1)
        
        # 2. Fuse with the heuristic scores if they are injected
        if heuristic_scores is not None:
            # Stack the neural output and the heuristic scalar side-by-side (Shape: [num_edges, 2])
            combined_features = torch.stack([neural_scores, heuristic_scores], dim=1)
            # Pass through the fusion layer to get the final logit
            final_scores = self.feature_fusion(combined_features).squeeze(-1)
            return final_scores
            
        return neural_scores

    def update_centroids_ema(self, z: torch.Tensor, q: torch.Tensor, momentum: float = 0.9):
        pred_labels = torch.argmax(q, dim=1)
        new_centers = self.cluster_centers.clone()
        for k in range(self.n_clusters):
            mask = (pred_labels == k)
            if mask.sum() > 0:
                c_k = z[mask].mean(dim=0)
                new_centers[k] = (momentum * self.cluster_centers[k]) + ((1.0 - momentum) * c_k)
            else:
                random_active_node = z[torch.randint(0, z.size(0), (1,))].squeeze(0)
                noise = torch.randn_like(random_active_node) * 0.05
                new_centers[k] = random_active_node + noise
        new_centers = F.normalize(new_centers, p=2, dim=1)
        self.cluster_centers.data.copy_(new_centers.data)