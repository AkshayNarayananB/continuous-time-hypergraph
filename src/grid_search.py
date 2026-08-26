import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from models.temphyper_dec import TempHyperDEC

# --- Helper Functions ---
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

def score_heuristic(edge, co_occurrence_tracker, node_frequency):
    if len(edge) < 2:
        return node_frequency.get(edge[0], 0)
    pairs = list(combinations(edge, 2))
    total_score = sum(co_occurrence_tracker.get(tuple(sorted([u, v])), 0) for u, v in pairs)
    return total_score / len(pairs)

def run_grid_search(dataset="email"):
    print(f"--- Starting OVERNIGHT HYPERPARAMETER GRID SEARCH on {dataset.upper()} ---")
    
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
    
    # THE GRID
    memory_dims = [64, 128, 256]
    hidden_dims = [32, 64]
    learning_rates = [0.001, 0.003]
    
    results_file = os.path.join(PROJECT_ROOT, "grid_search_results.csv")
    
    # Create or overwrite the CSV with headers
    with open(results_file, 'w') as f:
        f.write("Memory_Dim,Hidden_Dim,LR,ALL_AUC,NOVEL_AUC,AP\n")

    total_runs = len(memory_dims) * len(hidden_dims) * len(learning_rates)
    current_run = 1

    for mem_dim in memory_dims:
        for hid_dim in hidden_dims:
            for lr in learning_rates:
                print(f"\n[{current_run}/{total_runs}] Testing Config -> Mem: {mem_dim} | Hid: {hid_dim} | LR: {lr}")
                
                model = TempHyperDEC(num_nodes, memory_dim=mem_dim, time_dim=32, hidden_dim=hid_dim, n_clusters=2, use_attention=True).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                criterion = nn.BCEWithLogitsLoss()
                
                co_occurrence_tracker = defaultdict(int)
                node_frequency = defaultdict(int)
                seen_hyperedges = set()
                
                ap_scores, auc_scores, novel_auc_scores = [], [], []

                for t_idx, t in enumerate(time_steps):
                    pos_edges = df_edges[df_edges['time_step'] == t]['hyperedge_nodes'].tolist()
                    if not pos_edges: continue
                        
                    current_t_tensor = torch.tensor([t_idx], dtype=torch.float32, device=device)
                    neg_edges = generate_negative_hyperedges(pos_edges, num_nodes)
                    
                    pos_node_idx, pos_edge_idx = prepare_hyperedge_batch(pos_edges, device)
                    neg_node_idx, neg_edge_idx = prepare_hyperedge_batch(neg_edges, device)
                    
                    pos_heuristic = torch.tensor([score_heuristic(e, co_occurrence_tracker, node_frequency) for e in pos_edges], dtype=torch.float32, device=device)
                    neg_heuristic = torch.tensor([score_heuristic(e, co_occurrence_tracker, node_frequency) for e in neg_edges], dtype=torch.float32, device=device)
                    
                    novel_pos_mask = torch.tensor([e not in seen_hyperedges for e in pos_edges], dtype=torch.bool, device=device)
                    novel_neg_mask = torch.tensor([e not in seen_hyperedges for e in neg_edges], dtype=torch.bool, device=device)

                    if t_idx > 10:  
                        model.eval()
                        with torch.no_grad():
                            h = model.memory_module.get_memory() + model.node_emb.weight
                            z_pos_eval = F.normalize(model.hypergraph_layer(h, pos_node_idx, pos_edge_idx), p=2, dim=1)
                            z_neg_eval = F.normalize(model.hypergraph_layer(h, neg_node_idx, neg_edge_idx), p=2, dim=1)
                            
                            pos_scores_eval = model.predict_links(z_pos_eval, pos_node_idx, pos_edge_idx, len(pos_edges), heuristic_scores=pos_heuristic)
                            neg_scores_eval = model.predict_links(z_neg_eval, neg_node_idx, neg_edge_idx, len(neg_edges), heuristic_scores=neg_heuristic)
                            
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

                    model.train()
                    optimizer.zero_grad()
                    z_pos, _ = model(pos_node_idx, pos_edge_idx, current_t_tensor)
                    h_updated = model.memory_module.get_memory() + model.node_emb.weight
                    z_neg = F.normalize(model.hypergraph_layer(h_updated, neg_node_idx, neg_edge_idx), p=2, dim=1)
                    
                    pos_scores = model.predict_links(z_pos, pos_node_idx, pos_edge_idx, len(pos_edges), heuristic_scores=pos_heuristic)
                    neg_scores = model.predict_links(z_neg, neg_node_idx, neg_edge_idx, len(neg_edges), heuristic_scores=neg_heuristic)
                    
                    loss = criterion(pos_scores, torch.ones_like(pos_scores)) + criterion(neg_scores, torch.zeros_like(neg_scores))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    model.memory_module.memory.detach_()
                    
                    for edge in pos_edges:
                        seen_hyperedges.add(edge)
                        for node in edge: node_frequency[node] += 1
                        if len(edge) >= 2:
                            for u, v in combinations(edge, 2):
                                co_occurrence_tracker[tuple(sorted([u, v]))] += 1

                # Calculate final metrics for this config
                final_ap = np.mean(ap_scores)
                final_auc = np.mean(auc_scores)
                final_novel_auc = np.mean(novel_auc_scores)
                
                print(f"Result -> ALL AUC: {final_auc:.4f} | NOVEL AUC: {final_novel_auc:.4f}")
                
                # Append to CSV immediately
                with open(results_file, 'a') as f:
                    f.write(f"{mem_dim},{hid_dim},{lr},{final_auc:.4f},{final_novel_auc:.4f},{final_ap:.4f}\n")
                    
                current_run += 1

    print(f"\n================ GRID SEARCH COMPLETE ================")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    run_grid_search(dataset="email")