import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import scatter
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def generate_negative_hyperedges_batch(real_edges, num_nodes):
    """Vectorized-friendly negative hyperedge sampling."""
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


def prepare_hyperedge_pairs(hyperedges, device):
    """
    Flattens hyperedge combinations into 1D source/target node indices
    and edge-segment IDs for fast GPU scatter-mean reduction.
    """
    src_list, dst_list, seg_list = [], [], []
    for edge_idx, edge in enumerate(hyperedges):
        if len(edge) >= 2:
            pairs = list(combinations(edge, 2))
            u_nodes, v_nodes = zip(*pairs)
            src_list.extend(u_nodes)
            dst_list.extend(v_nodes)
            seg_list.extend([edge_idx] * len(pairs))

    if not src_list:
        return None

    src = torch.tensor(src_list, dtype=torch.long, device=device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    segment_ids = torch.tensor(seg_list, dtype=torch.long, device=device)
    return src, dst, segment_ids, len(hyperedges)


def prepare_memory_update_indices(edges, device):
    """
    Constructs symmetric pair interaction endpoints for batched bidirectional
    message aggregation via scatter_mean.
    """
    u_list, v_list = [], []
    for edge in edges:
        if len(edge) >= 2:
            for u, v in combinations(edge, 2):
                # Add bidirectional interactions
                u_list.extend([u, v])
                v_list.extend([v, u])

    if not u_list:
        return None

    target_nodes = torch.tensor(u_list, dtype=torch.long, device=device)
    source_nodes = torch.tensor(v_list, dtype=torch.long, device=device)
    return target_nodes, source_nodes


class TemporalPairwisePredictor(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        
        # Continuous-Time Node Memory (GRU state buffer)
        self.register_buffer("memory", torch.zeros(num_nodes, hidden_dim))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Simple Pairwise Link Predictor
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def get_representations(self):
        """Combines static node embeddings with dynamic GRU memory."""
        return self.memory + self.node_emb.weight
        
    def predict_hyperedges_fast(self, z, pair_info):
        """
        Batches all pairwise MLP forward passes across hyperedges
        and aggregates per hyperedge via scatter_mean.
        """
        if pair_info is None:
            return torch.zeros(0, device=z.device)

        src, dst, segment_ids, total_edges = pair_info
        
        # Vectorized pair feature concatenation
        pair_feats = torch.cat([z[src], z[dst]], dim=-1)
        pair_scores = self.link_predictor(pair_feats).squeeze(-1)
        
        # Scatter-mean reduction to get hyperedge scores
        agg_scores = scatter(pair_scores, segment_ids, dim=0, dim_size=total_edges, reduce='mean')
        return agg_scores

    def update_memory_fast(self, z, update_info):
        """
        Batched GRU memory update for all active interacting nodes.
        """
        if update_info is None:
            return

        target_nodes, source_nodes = update_info
        
        # Find unique active nodes being updated
        unique_targets, inverse_indices = torch.unique(target_nodes, return_inverse=True)
        
        # Vectorized scatter-mean message aggregation
        source_messages = z[source_nodes]
        agg_msgs = scatter(
            source_messages, 
            inverse_indices, 
            dim=0, 
            dim_size=len(unique_targets), 
            reduce='mean'
        )
        
        # Batched GRU step
        curr_mem = self.memory[unique_targets]
        new_mem = self.gru(agg_msgs, curr_mem)
        self.memory[unique_targets] = new_mem.detach()


def run_temporal_pairwise_baseline(dataset="mathoverflow"):
    print(f"--- Running OPTIMIZED CONTINUOUS-TIME PAIRWISE (CT-RNN) Baseline on {dataset.upper()} ---")
    
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
    
    model = TemporalPairwisePredictor(num_nodes, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    seen_hyperedges = set()
    ap_scores, auc_scores = [], []
    novel_auc_scores = []

    # Pre-group hyperedges by time step to prevent repeating dataframe filtering
    time_grouped_edges = {t: group['hyperedge_nodes'].tolist() for t, group in df_edges.groupby('time_step')}

    for t_idx, t in enumerate(time_steps):
        pos_edges = time_grouped_edges.get(t, [])
        if not pos_edges:
            continue
            
        neg_edges = generate_negative_hyperedges_batch(pos_edges, num_nodes)
        
        novel_pos = [e for e in pos_edges if e not in seen_hyperedges]
        novel_neg = [e for e in neg_edges if e not in seen_hyperedges]

        pos_info = prepare_hyperedge_pairs(pos_edges, device)
        neg_info = prepare_hyperedge_pairs(neg_edges, device)

        # ==========================================
        # PHASE 1: PREDICT (Batched Inference)
        # ==========================================
        if t_idx > 10:
            model.eval()
            with torch.inference_mode():
                z = model.get_representations()
                pos_scores = model.predict_hyperedges_fast(z, pos_info)
                neg_scores = model.predict_hyperedges_fast(z, neg_info)
                
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
        # PHASE 2: TRAIN & UPDATE MEMORY
        # ==========================================
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        z = model.get_representations()
        pos_scores = model.predict_hyperedges_fast(z, pos_info)
        neg_scores = model.predict_hyperedges_fast(z, neg_info)
        
        loss = criterion(pos_scores, torch.ones_like(pos_scores)) + \
               criterion(neg_scores, torch.zeros_like(neg_scores))
               
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track seen hyperedges
        seen_hyperedges.update(pos_edges)
        
        # Parallel GPU memory update
        mem_update_info = prepare_memory_update_indices(pos_edges, device)
        with torch.no_grad():
            z_detached = model.get_representations().detach()
            model.update_memory_fast(z_detached, mem_update_info)

    mean_ap = np.mean(ap_scores)
    mean_auc = np.mean(auc_scores)
    mean_novel_auc = np.mean(novel_auc_scores) if novel_auc_scores else float('nan')

    print("\n================ TEMPORAL PAIRWISE (CT-RNN) METRICS ================")
    print(f"Mean Average Precision (AP): {mean_ap:.4f}")
    print(f"Mean ROC-AUC (ALL): {mean_auc:.4f}")
    print(f"Mean ROC-AUC (NOVEL): {mean_novel_auc:.4f}")
    print("====================================================================")

    model_name = "temporal_pairwise_ctrnn"
    output_filename = f"{model_name}_{dataset}_metrics.txt"
    with open(output_filename, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Mean Average Precision (AP): {mean_ap:.4f}\n")
        f.write(f"Mean ROC-AUC (ALL): {mean_auc:.4f}\n")
        f.write(f"Mean ROC-AUC (NOVEL): {mean_novel_auc:.4f}\n")


if __name__ == "__main__":
    run_temporal_pairwise_baseline(dataset="email")
