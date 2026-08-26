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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
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

def score_heuristic(edge, co_occurrence_tracker, node_frequency):
    """Calculates the raw historical frequency scalar for a hyperedge."""
    if len(edge) < 2:
        return node_frequency.get(edge[0], 0)
    pairs = list(combinations(edge, 2))
    total_score = sum(co_occurrence_tracker.get(tuple(sorted([u, v])), 0) for u, v in pairs)
    return total_score / len(pairs)

def train_link_prediction(dataset="email"):
    print(f"--- Loading {dataset.upper()} Dataset for FEATURE-INJECTED LINK PREDICTION ---")
    
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    df_edges = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_edges.pkl"))
    df_labels = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_labels.pkl"))

    all_unique_nodes = set()
    for edges in df_edges['hyperedge_nodes']:
        all_unique_nodes.update(edges)
    all_unique_nodes.update(df_labels['node_id'].unique())
    node_mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted(all_unique_nodes))}
    df_edges['hyperedge_nodes'] = df_edges['hyperedge_nodes'].apply(lambda edge: tuple(sorted(node_mapping[n] for n in edge)))
    df_labels['node_id'] = df_labels['node_id'].map(node_mapping)
        
    num_nodes = len(node_mapping)
    time_steps = sorted(df_edges['time_step'].unique())
    
    MEMORY_DIM = 128
    TIME_DIM = 32
    HIDDEN_DIM = 64
    LR = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Nodes: {num_nodes} | Time Steps: {len(time_steps)}")
    
    model = TempHyperDEC(num_nodes, MEMORY_DIM, TIME_DIM, HIDDEN_DIM, n_clusters=2, use_attention=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # --- TRACKERS FOR FEATURE INJECTION ---
    co_occurrence_tracker = defaultdict(int)
    node_frequency = defaultdict(int)
    seen_hyperedges = set()
    
    tracked_steps = []
    ap_scores, auc_scores = [], []
    novel_auc_scores = []
    
    print("\n--- Starting Prequential (Predict-then-Train) Evaluation ---")

    for t_idx, t in enumerate(time_steps):
        pos_edges = df_edges[df_edges['time_step'] == t]['hyperedge_nodes'].tolist()
        if not pos_edges:
            continue
            
        current_t_tensor = torch.tensor([t_idx], dtype=torch.float32, device=device)
        neg_edges = generate_negative_hyperedges(pos_edges, num_nodes)
        
        pos_node_idx, pos_edge_idx = prepare_hyperedge_batch(pos_edges, device)
        neg_node_idx, neg_edge_idx = prepare_hyperedge_batch(neg_edges, device)
        
        # Calculate Heuristic Scalars for Injection
        pos_heuristic = torch.tensor([score_heuristic(e, co_occurrence_tracker, node_frequency) for e in pos_edges], dtype=torch.float32, device=device)
        neg_heuristic = torch.tensor([score_heuristic(e, co_occurrence_tracker, node_frequency) for e in neg_edges], dtype=torch.float32, device=device)
        
        # Track Novelty
        novel_pos_mask = torch.tensor([e not in seen_hyperedges for e in pos_edges], dtype=torch.bool, device=device)
        novel_neg_mask = torch.tensor([e not in seen_hyperedges for e in neg_edges], dtype=torch.bool, device=device)

        # ==========================================
        # PHASE 1: PREDICT 
        # ==========================================
        if t_idx > 10:  
            model.eval()
            with torch.no_grad():
                h = model.memory_module.get_memory() + model.node_emb.weight
                z_pos_eval = F.normalize(model.hypergraph_layer(h, pos_node_idx, pos_edge_idx), p=2, dim=1)
                z_neg_eval = F.normalize(model.hypergraph_layer(h, neg_node_idx, neg_edge_idx), p=2, dim=1)
                
                # INJECT FEATURES HERE
                pos_scores_eval = model.predict_links(z_pos_eval, pos_node_idx, pos_edge_idx, len(pos_edges), heuristic_scores=pos_heuristic)
                neg_scores_eval = model.predict_links(z_neg_eval, neg_node_idx, neg_edge_idx, len(neg_edges), heuristic_scores=neg_heuristic)
                
                y_true = np.concatenate([np.ones(len(pos_scores_eval)), np.zeros(len(neg_scores_eval))])
                y_scores = torch.cat([pos_scores_eval, neg_scores_eval]).cpu().numpy()
                
                if not np.isnan(y_scores).any():
                    ap = average_precision_score(y_true, y_scores)
                    auc = roc_auc_score(y_true, y_scores)
                    tracked_steps.append(t_idx)
                    ap_scores.append(ap)
                    auc_scores.append(auc)
                    
                    # Compute Novel Edge AUC
                    if novel_pos_mask.any() and novel_neg_mask.any():
                        n_y_true = np.concatenate([np.ones(novel_pos_mask.sum().item()), np.zeros(novel_neg_mask.sum().item())])
                        n_y_scores = torch.cat([pos_scores_eval[novel_pos_mask], neg_scores_eval[novel_neg_mask]]).cpu().numpy()
                        try:
                            novel_auc = roc_auc_score(n_y_true, n_y_scores)
                            novel_auc_scores.append(novel_auc)
                        except ValueError:
                            pass

                    if t_idx % 25 == 0:
                        nov_print = f"{novel_auc_scores[-1]:.4f}" if novel_auc_scores else "N/A"
                        print(f"Step {t_idx:03d} | ALL AUC: {auc:.4f} | NOVEL AUC: {nov_print}")

        # ==========================================
        # PHASE 2: TRAIN 
        # ==========================================
        model.train()
        optimizer.zero_grad()
        
        z_pos, _ = model(pos_node_idx, pos_edge_idx, current_t_tensor)
        h_updated = model.memory_module.get_memory() + model.node_emb.weight
        z_neg = F.normalize(model.hypergraph_layer(h_updated, neg_node_idx, neg_edge_idx), p=2, dim=1)
        
        # INJECT FEATURES HERE FOR GRADIENT DESCENT
        pos_scores = model.predict_links(z_pos, pos_node_idx, pos_edge_idx, len(pos_edges), heuristic_scores=pos_heuristic)
        neg_scores = model.predict_links(z_neg, neg_node_idx, neg_edge_idx, len(neg_edges), heuristic_scores=neg_heuristic)
        
        labels_pos = torch.ones_like(pos_scores)
        labels_neg = torch.zeros_like(neg_scores)
        
        loss_pos = criterion(pos_scores, labels_pos)
        loss_neg = criterion(neg_scores, labels_neg)
        loss = loss_pos + loss_neg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.memory_module.memory.detach_()
        
        # UPDATE TRACKERS
        for edge in pos_edges:
            seen_hyperedges.add(edge)
            for node in edge:
                node_frequency[node] += 1
            if len(edge) >= 2:
                for u, v in combinations(edge, 2):
                    pair = tuple(sorted([u, v]))
                    co_occurrence_tracker[pair] += 1

    print("\n================ FINAL PREDICTION METRICS ================")
    print(f"Mean Average Precision (AP): {np.mean(ap_scores):.4f}")
    print(f"Mean ROC-AUC (ALL): {np.mean(auc_scores):.4f}")
    print(f"Mean ROC-AUC (NOVEL): {np.mean(novel_auc_scores):.4f}")
    print("==========================================================")

if __name__ == "__main__":
    train_link_prediction(dataset="email")