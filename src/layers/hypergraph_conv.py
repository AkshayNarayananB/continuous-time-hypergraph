import torch
import torch.nn as nn

class HypergraphConv(nn.Module):
    """
    Dual-stage sparse hypergraph convolution layer using NATIVE PyTorch.
    Transforms node features by routing them through hyperedges and back.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Learnable transformation matrices for nodes and edges
        self.node_transform = nn.Linear(in_channels, out_channels)
        self.edge_transform = nn.Linear(out_channels, out_channels)
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor, node_indices: torch.Tensor, hyperedge_indices: torch.Tensor):
        """
        x: [num_nodes, in_channels] - The current node features (e.g., memory states)
        node_indices: 1D tensor of nodes involved in current batch of edges
        hyperedge_indices: 1D tensor mapping each node to its corresponding hyperedge ID
        """
        # Step 0: Initial node feature transformation
        x_transformed = self.node_transform(x)
        
        # Step 1: Node -> Hyperedge Aggregation
        gathered_node_feats = x_transformed[node_indices]
        
        # Determine how many unique hyperedges exist in this batch
        num_active_edges = int(hyperedge_indices.max().item()) + 1
        
        # Expand indices to match feature dimensions for native PyTorch scatter
        idx_edge = hyperedge_indices.unsqueeze(1).expand_as(gathered_node_feats)
        
        # Initialize empty edge features and apply native scatter reduce (mean)
        edge_feats = torch.zeros(num_active_edges, x_transformed.size(1), device=x.device, dtype=x.dtype)
        edge_feats.scatter_reduce_(0, idx_edge, gathered_node_feats, reduce='mean', include_self=False)
        
        # Process edge features
        edge_feats = self.activation(self.edge_transform(edge_feats))
        
        # Step 2: Hyperedge -> Node Broadcast
        broadcasted_edge_feats = edge_feats[hyperedge_indices]
        
        # Expand node indices to match feature dimensions
        idx_node = node_indices.unsqueeze(1).expand_as(broadcasted_edge_feats)
        
        # Initialize empty node features and apply native scatter reduce back to nodes
        out_x = torch.zeros_like(x_transformed)
        out_x.scatter_reduce_(0, idx_node, broadcasted_edge_feats, reduce='mean', include_self=False)
        
        # Residual connection + activation
        out_x = self.activation(out_x + x_transformed)
        return out_x

class HypergraphAttention(nn.Module):
    """
    Dual-stage sparse hypergraph ATTENTION layer (HyperGAT).
    Learns dynamic weights to filter out passive observers in massive group interactions.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.node_transform = nn.Linear(in_channels, out_channels)
        self.edge_transform = nn.Linear(out_channels, out_channels)
        
        # The attention scoring mechanisms
        self.att_node_to_edge = nn.Linear(2 * out_channels, 1, bias=False)
        self.att_edge_to_node = nn.Linear(2 * out_channels, 1, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor, node_indices: torch.Tensor, hyperedge_indices: torch.Tensor):
        num_active_edges = int(hyperedge_indices.max().item()) + 1
        
        # Step 0: Base feature projection
        h = self.node_transform(x)
        gathered_h = h[node_indices]
        
        # --- PHASE 1: Node -> Edge Attention ---
        # 1a. Build a rough "context" of the edge using simple mean-pooling
        idx_edge = hyperedge_indices.unsqueeze(1).expand_as(gathered_h)
        edge_context = torch.zeros(num_active_edges, h.size(1), device=x.device, dtype=x.dtype)
        edge_context.scatter_reduce_(0, idx_edge, gathered_h, reduce='mean', include_self=False)
        
        # 1b. Concatenate node features with their edge context
        broadcasted_edge_ctx = edge_context[hyperedge_indices]
        cat_ne = torch.cat([gathered_h, broadcasted_edge_ctx], dim=-1)
        
        # 1c. Calculate unnormalized attention scores and apply Softmax over the edge group
        scores_ne = self.leaky_relu(self.att_node_to_edge(cat_ne))
        exp_scores_ne = torch.exp(scores_ne)
        
        sum_exp_ne = torch.zeros(num_active_edges, 1, device=x.device)
        sum_exp_ne.scatter_add_(0, hyperedge_indices.unsqueeze(1), exp_scores_ne)
        
        # Normalize to get the alpha weights [0, 1]
        alpha_ne = exp_scores_ne / (sum_exp_ne[hyperedge_indices] + 1e-9)
        
        # 1d. Apply weights and aggregate into final edge features
        weighted_h = gathered_h * alpha_ne
        edge_feats = torch.zeros(num_active_edges, h.size(1), device=x.device)
        edge_feats.scatter_add_(0, idx_edge, weighted_h)
        edge_feats = self.activation(self.edge_transform(edge_feats))
        
        # --- PHASE 2: Edge -> Node Attention ---
        broadcasted_edge_feats = edge_feats[hyperedge_indices]
        
        # 2a. Concatenate the finalized edge features with the original node features
        cat_en = torch.cat([broadcasted_edge_feats, gathered_h], dim=-1)
        
        # 2b. Calculate unnormalized attention scores and apply Softmax over the node group
        scores_en = self.leaky_relu(self.att_edge_to_node(cat_en))
        exp_scores_en = torch.exp(scores_en)
        
        sum_exp_en = torch.zeros(x.size(0), 1, device=x.device)
        sum_exp_en.scatter_add_(0, node_indices.unsqueeze(1), exp_scores_en)
        
        alpha_en = exp_scores_en / (sum_exp_en[node_indices] + 1e-9)
        
        # 2c. Apply weights and broadcast back to the global node embeddings
        weighted_edge = broadcasted_edge_feats * alpha_en
        idx_node = node_indices.unsqueeze(1).expand_as(weighted_edge)
        
        out_x = torch.zeros_like(h)
        out_x.scatter_add_(0, idx_node, weighted_edge)
        
        # Residual connection
        out_x = self.activation(out_x + h)
        return out_x