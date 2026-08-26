import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import scatter
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def generate_negative_hyperedges_fast(real_edges, num_nodes):
    """Vectorized negative hyperedge generation."""
    neg_edges = []
    rand_nodes = np.random.randint(0, num_nodes, size=len(real_edges))
    for i, edge in enumerate(real_edges):
        if not edge:
            neg_edges.append(tuple())
            continue
        neg_edge = list(edge)
        idx_to_replace = np.random.randint(0, len(neg_edge))
        neg_edge[idx_to_replace] = int(rand_nodes[i])
        neg_edges.append(tuple(sorted(set(neg_edge))))
    return neg_edges


def prepare_hyperedges_for_pooling(hyperedges, device):
    """
    Flattens hyperedge node membership into flat index tensors and segment maps
    for vectorized GPU scatter mean-pooling and scoring.
    """
    node_indices = []
    segment_ids = []
    
    for edge_idx, edge in enumerate(hyperedges):
        if len(edge) > 0:
            node_indices.extend(edge)
            segment_ids.extend([edge_idx] * len(edge))

    if not node_indices:
        return None

    nodes = torch.tensor(node_indices, dtype=torch.long, device=device)
    segments = torch.tensor(segment_ids, dtype=torch.long, device=device)
    return nodes, segments, len(hyperedges)


class StaticHGNNPredictor(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.conv1 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim)
        
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, hyperedge_index):
        x = self.node_emb.weight
        if hyperedge_index.size(1) > 0:
            x = F.relu(self.conv1(x, hyperedge_index))
            x = self.conv2(x, hyperedge_index)
        return x
        
    def predict_hyperedges_fast(self, z, batch_info):
        """
        Vectorized hyperedge scoring on GPU via scatter-mean pooling
        followed by a batched MLP forward pass.
        """
        if batch_info is None:
            return torch.zeros(0, device=z.device)

        nodes, segments, total_edges = batch_info
        
        # Parallel GPU mean-pooling of node representations per hyperedge
        node_embeddings = z[nodes]
        pooled = scatter(node_embeddings, segments, dim=0, dim_size=total_edges, reduce='mean')
        
        # Batched MLP prediction for all hyperedges at once
        scores = self.link_predictor(pooled).squeeze(-1)
        return scores


def run_hgnn_baseline(dataset="mathoverflow"):
    print(f"--- Running OPTIMIZED STATIC HYPERGRAPH (HGNN) Baseline on {dataset.upper()} ---")
    
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    df_edges = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_edges.pkl"))
    df_labels = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_labels.pkl"))

    all_unique_nodes = set()
    for edges in df_edges['hyperedge_nodes']:
        all_unique_nodes.update(edges)
    all_unique_nodes.update(df_labels['node_id'].unique())
    node_mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted(all_unique_nodes))}
    df_edges['hyperedge_nodes'] = df_edges['hyperedge_nodes'].apply(
        lambda edge: tuple(sorted(node_mapping[n] for n in edge))
    )
        
    num_nodes = len(node_mapping)
    time_steps = sorted(df_edges['time_step'].unique())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    
    model = StaticHGNNPredictor(num_nodes, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    # Incremental buffers for historical hypergraph indices
    hist_nodes_list = []
    hist_edges_list = []
    current_edge_count = 0
    seen_hyperedges = set()
    
    ap_scores, auc_scores = [], []
    novel_auc_scores = []

    # Pre-group hyperedges by time step to prevent repeating Pandas dataframe scans
    time_grouped_edges = {t: group['hyperedge_nodes'].tolist() for t, group in df_edges.groupby('time_step')}

    for t_idx, t in enumerate(time_steps):
        pos_edges = time_grouped_edges.get(t, [])
        if not pos_edges:
            continue
            
        neg_edges = generate_negative_hyperedges_fast(pos_edges, num_nodes)
        novel_pos = [e for e in pos_edges if e not in seen_hyperedges]
        novel_neg = [e for e in neg_edges if e not in seen_hyperedges]

        # Fast non-blocking hyperedge_index construction
        if hist_nodes_list:
            nodes_tensor = torch.tensor(hist_nodes_list, dtype=torch.long, device=device)
            edges_tensor = torch.tensor(hist_edges_list, dtype=torch.long, device=device)
            hyperedge_index = torch.stack([nodes_tensor, edges_tensor], dim=0)
        else:
            hyperedge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        pos_batch_info = prepare_hyperedges_for_pooling(pos_edges, device)
        neg_batch_info = prepare_hyperedges_for_pooling(neg_edges, device)

        # ==========================================
        # PHASE 1: PREDICT (Batched Inference)
        # ==========================================
        if t_idx > 10:
            model.eval()
            with torch.inference_mode():
                z = model(hyperedge_index)
                pos_scores = model.predict_hyperedges_fast(z, pos_batch_info)
                neg_scores = model.predict_hyperedges_fast(z, neg_batch_info)
                
                y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
                y_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                
                if not np.isnan(y_scores).any():
                    ap_scores.append(average_precision_score(y_true, y_scores))
                    auc_scores.append(roc_auc_score(y_true, y_scores))
                    
                    if novel_pos and novel_neg:
                        novel_pos_info = prepare_hyperedges_for_pooling(novel_pos, device)
                        novel_neg_info = prepare_hyperedges_for_pooling(novel_neg, device)
                        n_pos = model.predict_hyperedges_fast(z, novel_pos_info)
                        n_neg = model.predict_hyperedges_fast(z, novel_neg_info)
                        
                        n_true = np.concatenate([np.ones(len(n_pos)), np.zeros(len(n_neg))])
                        n_scores = torch.cat([n_pos, n_neg]).cpu().numpy()
                        try:
                            novel_auc_scores.append(roc_auc_score(n_true, n_scores))
                        except ValueError:
                            pass
                            
                    if t_idx % 25 == 0:
                        nov_print = f"{novel_auc_scores[-1]:.4f}" if novel_auc_scores else "N/A"
                        print(f"Step {t_idx:03d} | ALL AUC: {auc_scores[-1]:.4f} | NOVEL AUC: {nov_print}")

        # ==========================================
        # PHASE 2: TRAIN (Single Forward & Backward)
        # ==========================================
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model(hyperedge_index)
        
        pos_scores = model.predict_hyperedges_fast(z, pos_batch_info)
        neg_scores = model.predict_hyperedges_fast(z, neg_batch_info)
        
        loss = criterion(pos_scores, torch.ones_like(pos_scores)) + \
               criterion(neg_scores, torch.zeros_like(neg_scores))
               
        loss.backward()
        optimizer.step()
        
        # Accumulate the graph incrementally without rebuilding from scratch
        for edge in pos_edges:
            seen_hyperedges.add(edge)
            for node in edge:
                hist_nodes_list.append(node)
                hist_edges_list.append(current_edge_count)
            current_edge_count += 1

    mean_ap = np.mean(ap_scores)
    mean_auc = np.mean(auc_scores)
    mean_novel_auc = np.mean(novel_auc_scores) if novel_auc_scores else float('nan')

    print("\n================ STATIC HGNN METRICS ================")
    print(f"Mean Average Precision (AP): {mean_ap:.4f}")
    print(f"Mean ROC-AUC (ALL): {mean_auc:.4f}")
    print(f"Mean ROC-AUC (NOVEL): {mean_novel_auc:.4f}")
    print("=====================================================")

    model_name = "static_hgnn"
    output_filename = f"{model_name}_{dataset}_metrics.txt"
    with open(output_filename, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Mean Average Precision (AP): {mean_ap:.4f}\n")
        f.write(f"Mean ROC-AUC (ALL): {mean_auc:.4f}\n")
        f.write(f"Mean ROC-AUC (NOVEL): {mean_novel_auc:.4f}\n")


if __name__ == "__main__":
    run_hgnn_baseline(dataset="email")
