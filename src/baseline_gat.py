import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def generate_negative_hyperedges_batch(real_edges, num_nodes):
    """Vectorized-friendly generation of negative hyperedges."""
    neg_edges = []
    # Pre-generate random replacements
    rand_nodes = np.random.randint(0, num_nodes, size=len(real_edges))
    for i, edge in enumerate(real_edges):
        if len(edge) == 0:
            neg_edges.append(tuple())
            continue
        neg_edge = list(edge)
        idx_to_replace = np.random.randint(0, len(neg_edge))
        neg_edge[idx_to_replace] = int(rand_nodes[i])
        neg_edges.append(tuple(sorted(set(neg_edge))))
    return neg_edges


def prepare_hyperedge_pairs(hyperedges, device):
    """
    Flattens all hyperedge pairwise combinations into a single batch 
    of source/target indices and an edge-segment index for fast GPU reduction.
    """
    src_list, dst_list, segment_ids = [], [], []
    valid_edge_mask = []

    for edge_idx, edge in enumerate(hyperedges):
        if len(edge) >= 2:
            pairs = list(combinations(edge, 2))
            u_nodes, v_nodes = zip(*pairs)
            src_list.extend(u_nodes)
            dst_list.extend(v_nodes)
            segment_ids.extend([edge_idx] * len(pairs))
            valid_edge_mask.append(True)
        else:
            valid_edge_mask.append(False)

    if not src_list:
        return None

    src = torch.tensor(src_list, dtype=torch.long, device=device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
    valid_mask = torch.tensor(valid_edge_mask, dtype=torch.bool, device=device)
    
    return src, dst, segment_ids, valid_mask, len(hyperedges)


class StaticGATLinkPredictor(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        
    def forward(self, edge_index):
        x = self.node_emb.weight
        if edge_index.size(1) > 0:
            x = F.relu(self.gat1(x, edge_index))
            x = self.gat2(x, edge_index)
        return x
        
    def predict_hyperedges_fast(self, z, edge_batch_info):
        """
        Vectorized hyperedge scoring on GPU.
        Computes pairwise dot-products and aggregates via scatter_mean.
        """
        if edge_batch_info is None:
            return torch.zeros(0, device=z.device)
            
        src, dst, segment_ids, valid_mask, total_edges = edge_batch_info
        
        # Batched dot-product of all constituent pairs in parallel
        pair_scores = (z[src] * z[dst]).sum(dim=-1)
        
        # Scatter-mean reduction to aggregate pairs back into hyperedge scores
        scores = torch.zeros(total_edges, device=z.device)
        agg_scores = scatter(pair_scores, segment_ids, dim=0, dim_size=total_edges, reduce='mean')
        
        return agg_scores


def run_gat_baseline(dataset="mathoverflow"):
    print(f"--- Running OPTIMIZED STATIC PAIRWISE GAT on {dataset.upper()} ---")
    
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
    
    # Enable TF32 matrix multiplication on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    
    model = StaticGATLinkPredictor(num_nodes, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    historical_pairwise_edges = set()
    seen_hyperedges = set()
    
    ap_scores, auc_scores = [], []
    novel_auc_scores = []

    # Pre-group hyperedges by time step to prevent repeated DataFrame filtering
    time_grouped_edges = {t: group['hyperedge_nodes'].tolist() for t, group in df_edges.groupby('time_step')}

    for t_idx, t in enumerate(time_steps):
        pos_edges = time_grouped_edges.get(t, [])
        if not pos_edges:
            continue
            
        neg_edges = generate_negative_hyperedges_batch(pos_edges, num_nodes)
        
        novel_pos = [e for e in pos_edges if e not in seen_hyperedges]
        novel_neg = [e for e in neg_edges if e not in seen_hyperedges]

        # Convert historical pairwise edges directly into tensors
        if historical_pairwise_edges:
            edges_array = np.array(list(historical_pairwise_edges), dtype=np.int64)
            src = torch.from_numpy(edges_array[:, 0]).to(device)
            dst = torch.from_numpy(edges_array[:, 1]).to(device)
            edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        pos_batch_info = prepare_hyperedge_pairs(pos_edges, device)
        neg_batch_info = prepare_hyperedge_pairs(neg_edges, device)

        # ==========================================
        # PHASE 1: PREDICT (Batched Inference)
        # ==========================================
        if t_idx > 10:
            model.eval()
            with torch.inference_mode():
                z = model(edge_index)
                pos_scores = model.predict_hyperedges_fast(z, pos_batch_info)
                neg_scores = model.predict_hyperedges_fast(z, neg_batch_info)
                
                y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
                y_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                
                if not np.isnan(y_scores).any():
                    ap_scores.append(average_precision_score(y_true, y_scores))
                    auc_scores.append(roc_auc_score(y_true, y_scores))
                    
                    if novel_pos and novel_neg:
                        novel_pos_info = prepare_hyperedge_pairs(novel_pos, device)
                        novel_neg_info = prepare_hyperedge_pairs(novel_neg, device)
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
        # PHASE 2: TRAIN (Single GAT Forward & Backward)
        # ==========================================
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model(edge_index)
        
        pos_scores = model.predict_hyperedges_fast(z, pos_batch_info)
        neg_scores = model.predict_hyperedges_fast(z, neg_batch_info)
        
        loss = criterion(pos_scores, torch.ones_like(pos_scores)) + \
               criterion(neg_scores, torch.zeros_like(neg_scores))
               
        loss.backward()
        optimizer.step()
        
        # Batch add seen hyperedges & clique expansions
        seen_hyperedges.update(pos_edges)
        for edge in pos_edges:
            if len(edge) >= 2:
                for u, v in combinations(edge, 2):
                    historical_pairwise_edges.add((u, v) if u < v else (v, u))

    mean_ap = np.mean(ap_scores)
    mean_auc = np.mean(auc_scores)
    mean_novel_auc = np.mean(novel_auc_scores) if novel_auc_scores else float('nan')

    print("\n================ STATIC GAT METRICS ================")
    print(f"Mean Average Precision (AP): {mean_ap:.4f}")
    print(f"Mean ROC-AUC (ALL): {mean_auc:.4f}")
    print(f"Mean ROC-AUC (NOVEL): {mean_novel_auc:.4f}")
    print("====================================================")

   # Open the file in write mode
    with open("metrics.txt", "w") as f:
        f.write("\n================ STATIC HGNN METRICS ================\n")
        f.write(f"Mean Average Precision (AP): {mean_ap:.4f}\n")
        f.write(f"Mean ROC-AUC (ALL): {mean_auc:.4f}\n")
        f.write(f"Mean ROC-AUC (NOVEL): {mean_novel_auc:.4f}\n")
        f.write("=====================================================\n")

if __name__ == "__main__":
    run_gat_baseline(dataset="email")
