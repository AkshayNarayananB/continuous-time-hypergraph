import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from models.temphyper_dec import TempHyperDEC

def prepare_hyperedge_batch(batch_edges, device):
    node_idx, edge_idx = [], []
    for e_id, edge in enumerate(batch_edges):
        for node in edge:
            node_idx.append(node)
            edge_idx.append(e_id)
    return torch.tensor(node_idx, dtype=torch.long, device=device), \
           torch.tensor(edge_idx, dtype=torch.long, device=device)

def generate_negative_hyperedges(real_edges, num_nodes):
    neg_edges = []
    for edge in real_edges:
        neg_edge = list(edge)
        if len(neg_edge) > 0:
            idx_to_replace = random.randint(0, len(neg_edge) - 1)
            neg_edge[idx_to_replace] = random.randint(0, num_nodes - 1)
        neg_edges.append(tuple(set(neg_edge))) 
    return neg_edges

def run_ablation(dataset="email"):
    print(f"--- Running ABLATION: TempHyper (NO FEATURE INJECTION) on {dataset.upper()} ---")
    
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    df_edges = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_edges.pkl"))
    df_labels = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_labels.pkl"))

    all_unique_nodes = set()
    for edges in df_edges['hyperedge_nodes']:
        all_unique_nodes.update(edges)
    all_unique_nodes.update(df_labels['node_id'].unique())
    node_mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted(all_unique_nodes))}
    df_edges['hyperedge_nodes'] = df_edges['hyperedge_nodes'].apply(lambda edge: tuple(sorted(node_mapping[n] for n in edge)))
        
    num_nodes = len(node_mapping)
    time_steps = sorted(df_edges['time_step'].unique())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = TempHyperDEC(num_nodes, memory_dim=128, time_dim=32, hidden_dim=64, n_clusters=2, use_attention=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    seen_hyperedges = set()
    ap_scores, auc_scores, novel_auc_scores = [], [], []

    for t_idx, t in enumerate(time_steps):
        pos_edges = df_edges[df_edges['time_step'] == t]['hyperedge_nodes'].tolist()
        if not pos_edges: continue
            
        current_t_tensor = torch.tensor([t_idx], dtype=torch.float32, device=device)
        neg_edges = generate_negative_hyperedges(pos_edges, num_nodes)
        
        pos_node_idx, pos_edge_idx = prepare_hyperedge_batch(pos_edges, device)
        neg_node_idx, neg_edge_idx = prepare_hyperedge_batch(neg_edges, device)
        
        novel_pos_mask = torch.tensor([e not in seen_hyperedges for e in pos_edges], dtype=torch.bool, device=device)
        novel_neg_mask = torch.tensor([e not in seen_hyperedges for e in neg_edges], dtype=torch.bool, device=device)

        # ==========================================
        # PHASE 1: PREDICT (Pure Neural Only)
        # ==========================================
        if t_idx > 10:  
            model.eval()
            with torch.no_grad():
                h = model.memory_module.get_memory() + model.node_emb.weight
                z_pos_eval = F.normalize(model.hypergraph_layer(h, pos_node_idx, pos_edge_idx), p=2, dim=1)
                z_neg_eval = F.normalize(model.hypergraph_layer(h, neg_node_idx, neg_edge_idx), p=2, dim=1)
                
                # NO HEURISTIC INJECTED HERE
                pos_scores_eval = model.predict_links(z_pos_eval, pos_node_idx, pos_edge_idx, len(pos_edges))
                neg_scores_eval = model.predict_links(z_neg_eval, neg_node_idx, neg_edge_idx, len(neg_edges))
                
                y_true = np.concatenate([np.ones(len(pos_scores_eval)), np.zeros(len(neg_scores_eval))])
                y_scores = torch.cat([pos_scores_eval, neg_scores_eval]).cpu().numpy()
                
                if not np.isnan(y_scores).any():
                    ap_scores.append(average_precision_score(y_true, y_scores))
                    auc_scores.append(roc_auc_score(y_true, y_scores))
                    
                    if novel_pos_mask.any() and novel_neg_mask.any():
                        n_y_true = np.concatenate([np.ones(novel_pos_mask.sum().item()), np.zeros(novel_neg_mask.sum().item())])
                        n_y_scores = torch.cat([pos_scores_eval[novel_pos_mask], neg_scores_eval[novel_neg_mask]]).cpu().numpy()
                        try:
                            novel_auc_scores.append(roc_auc_score(n_y_true, n_y_scores))
                        except ValueError:
                            pass
                    
                    if t_idx % 50 == 0:
                        nov_print = f"{novel_auc_scores[-1]:.4f}" if novel_auc_scores else "N/A"
                        print(f"Step {t_idx:03d} | ALL AUC: {auc_scores[-1]:.4f} | NOVEL AUC: {nov_print}")

        # ==========================================
        # PHASE 2: TRAIN (Pure Neural Only)
        # ==========================================
        model.train()
        optimizer.zero_grad()
        
        z_pos, _ = model(pos_node_idx, pos_edge_idx, current_t_tensor)
        h_updated = model.memory_module.get_memory() + model.node_emb.weight
        z_neg = F.normalize(model.hypergraph_layer(h_updated, neg_node_idx, neg_edge_idx), p=2, dim=1)
        
        # NO HEURISTIC INJECTED HERE
        pos_scores = model.predict_links(z_pos, pos_node_idx, pos_edge_idx, len(pos_edges))
        neg_scores = model.predict_links(z_neg, neg_node_idx, neg_edge_idx, len(neg_edges))
        
        loss = criterion(pos_scores, torch.ones_like(pos_scores)) + \
               criterion(neg_scores, torch.zeros_like(neg_scores))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.memory_module.memory.detach_()
        
        for edge in pos_edges:
            seen_hyperedges.add(edge)

    print("\n================ ABLATION METRICS (NO FUSION) ================")
    print(f"Mean Average Precision (AP): {np.mean(ap_scores):.4f}")
    print(f"Mean ROC-AUC (ALL): {np.mean(auc_scores):.4f}")
    print(f"Mean ROC-AUC (NOVEL): {np.mean(novel_auc_scores):.4f}")
    print("==============================================================")

if __name__ == "__main__":
    run_ablation(dataset="email")