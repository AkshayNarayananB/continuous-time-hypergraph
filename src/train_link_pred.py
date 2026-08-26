import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import scatter
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from models.temphyper_dec import TempHyperDEC


def generate_negative_hyperedges_fast(real_edges, num_nodes):
    """Vectorized negative hyperedge replacement."""
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


def prepare_hyperedge_batch_fast(batch_edges, device):
    """Generates PyG hyperedge incidence indices in a contiguous tensor format."""
    node_idx, edge_idx = [], []
    for e_id, edge in enumerate(batch_edges):
        for node in edge:
            node_idx.append(node)
            edge_idx.append(e_id)
            
    if not node_idx:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
        
    return torch.tensor(node_idx, dtype=torch.long, device=device), \
           torch.tensor(edge_idx, dtype=torch.long, device=device)


def prepare_heuristic_tensors(hyperedges, device):
    """Prepares pair indices and singletons for vectorized GPU scoring."""
    pair_u, pair_v, pair_seg = [], [], []
    single_nodes, single_seg = [], []
    pair_counts = np.zeros(len(hyperedges), dtype=np.float32)

    for edge_idx, edge in enumerate(hyperedges):
        if len(edge) >= 2:
            pairs = list(combinations(edge, 2))
            u_nodes, v_nodes = zip(*pairs)
            pair_u.extend(u_nodes)
            pair_v.extend(v_nodes)
            pair_seg.extend([edge_idx] * len(pairs))
            pair_counts[edge_idx] = float(len(pairs))
        elif len(edge) == 1:
            single_nodes.append(edge[0])
            single_seg.append(edge_idx)

    pair_info = None
    if pair_u:
        pair_info = (
            torch.tensor(pair_u, dtype=torch.long, device=device),
            torch.tensor(pair_v, dtype=torch.long, device=device),
            torch.tensor(pair_seg, dtype=torch.long, device=device),
        )

    single_info = None
    if single_nodes:
        single_info = (
            torch.tensor(single_nodes, dtype=torch.long, device=device),
            torch.tensor(single_seg, dtype=torch.long, device=device),
        )

    counts = torch.from_numpy(pair_counts).to(device)
    return pair_info, single_info, counts, len(hyperedges)


class GPUHeuristicEngine:
    """GPU-accelerated pairwise co-occurrence and node frequency bank."""
    def __init__(self, num_nodes, device):
        self.num_nodes = num_nodes
        self.device = device
        self.node_frequency = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.co_occurrence_matrix = torch.sparse_coo_tensor(
            size=(num_nodes, num_nodes), dtype=torch.float32, device=device
        )
        self.cached_co_dense = None

    def update(self, pos_edges):
        all_nodes = [node for edge in pos_edges for node in edge]
        if all_nodes:
            nodes_tensor = torch.tensor(all_nodes, dtype=torch.long, device=self.device)
            self.node_frequency.scatter_add_(
                0, nodes_tensor, torch.ones_like(nodes_tensor, dtype=torch.float32)
            )

        pairs_u, pairs_v = [], []
        for edge in pos_edges:
            if len(edge) >= 2:
                for u, v in combinations(edge, 2):
                    pairs_u.append(u if u < v else v)
                    pairs_v.append(v if u < v else u)

        if pairs_u:
            indices = torch.tensor([pairs_u, pairs_v], dtype=torch.long, device=self.device)
            values = torch.ones(len(pairs_u), dtype=torch.float32, device=self.device)
            new_co = torch.sparse_coo_tensor(
                indices, values, size=(self.num_nodes, self.num_nodes), device=self.device
            )
            self.co_occurrence_matrix = (self.co_occurrence_matrix + new_co).coalesce()
            self.cached_co_dense = None

    def _get_dense_view(self):
        if self.cached_co_dense is None:
            self.cached_co_dense = self.co_occurrence_matrix.to_dense()
        return self.cached_co_dense

    def score(self, heuristic_batch_data):
        if heuristic_batch_data is None:
            return torch.zeros(0, device=self.device)

        pair_info, single_info, counts, total_edges = heuristic_batch_data
        scores = torch.zeros(total_edges, dtype=torch.float32, device=self.device)

        if pair_info is not None:
            u, v, seg_ids = pair_info
            dense_co = self._get_dense_view()
            pair_values = dense_co[u, v]
            summed_scores = scatter(pair_values, seg_ids, dim=0, dim_size=total_edges, reduce='sum')
            scores = scores + torch.where(counts > 0, summed_scores / torch.clamp(counts, min=1.0), 0.0)

        if single_info is not None:
            nodes, seg_ids = single_info
            scores[seg_ids] = self.node_frequency[nodes]

        return scores


def train_link_prediction(dataset="email"):
    print(f"--- Loading {dataset.upper()} Dataset for MASTER LINK PREDICTION ---")
    
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
    df_labels['node_id'] = df_labels['node_id'].map(node_mapping)
        
    num_nodes = len(node_mapping)
    time_steps = sorted(df_edges['time_step'].unique())
    
    MEMORY_DIM = 64
    TIME_DIM = 32
    HIDDEN_DIM = 32
    LR = 0.002
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        
    print(f"Nodes: {num_nodes} | Time Steps: {len(time_steps)} | Device: {device}")
    
    model = TempHyperDEC(num_nodes, MEMORY_DIM, TIME_DIM, HIDDEN_DIM, n_clusters=2, use_attention=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) 
    criterion = nn.BCEWithLogitsLoss()
    
    heuristic_engine = GPUHeuristicEngine(num_nodes, device)
    
    print("\n--- Starting Prequential (Predict-then-Train) Evaluation ---")
    
    tracked_steps = []
    ap_scores, auc_scores = [], []

    # Pre-aggregate timesteps to eliminate Pandas row-filtering overhead
    time_grouped_edges = {t: group['hyperedge_nodes'].tolist() for t, group in df_edges.groupby('time_step')}

    for t_idx, t in enumerate(time_steps):
        pos_edges = time_grouped_edges.get(t, [])
        if not pos_edges:
            continue
            
        current_t_tensor = torch.tensor([t_idx], dtype=torch.float32, device=device)
        neg_edges = generate_negative_hyperedges_fast(pos_edges, num_nodes)
        
        pos_node_idx, pos_edge_idx = prepare_hyperedge_batch_fast(pos_edges, device)
        neg_node_idx, neg_edge_idx = prepare_hyperedge_batch_fast(neg_edges, device)
        
        pos_h_data = prepare_heuristic_tensors(pos_edges, device)
        neg_h_data = prepare_heuristic_tensors(neg_edges, device)
        
        pos_heuristic = heuristic_engine.score(pos_h_data)
        neg_heuristic = heuristic_engine.score(neg_h_data)
        
        # ==========================================
        # PHASE 1: PREDICT (Vectorized Evaluation)
        # ==========================================
        if t_idx > 10:  
            model.eval()
            with torch.inference_mode():
                h = model.memory_module.get_memory() + model.node_emb.weight
                z_pos_eval = F.normalize(model.hypergraph_layer(h, pos_node_idx, pos_edge_idx), p=2, dim=1)
                z_neg_eval = F.normalize(model.hypergraph_layer(h, neg_node_idx, neg_edge_idx), p=2, dim=1)
                
                pos_scores_eval = model.predict_links(
                    z_pos_eval, pos_node_idx, pos_edge_idx, len(pos_edges), heuristic_scores=pos_heuristic
                )
                neg_scores_eval = model.predict_links(
                    z_neg_eval, neg_node_idx, neg_edge_idx, len(neg_edges), heuristic_scores=neg_heuristic
                )
                
                y_true = np.concatenate([np.ones(len(pos_scores_eval)), np.zeros(len(neg_scores_eval))])
                y_scores = torch.cat([pos_scores_eval, neg_scores_eval]).cpu().numpy()
                
                if not np.isnan(y_scores).any():
                    ap = average_precision_score(y_true, y_scores)
                    auc = roc_auc_score(y_true, y_scores)
                    
                    tracked_steps.append(t_idx)
                    ap_scores.append(ap)
                    auc_scores.append(auc)
                    
                    if t_idx % 10 == 0:
                        print(f"Step {t_idx:03d} | AP: {ap:.4f} | ROC-AUC: {auc:.4f}")

        # ==========================================
        # PHASE 2: TRAIN
        # ==========================================
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        z_pos, _ = model(pos_node_idx, pos_edge_idx, current_t_tensor)
        h_updated = model.memory_module.get_memory() + model.node_emb.weight
        z_neg = F.normalize(model.hypergraph_layer(h_updated, neg_node_idx, neg_edge_idx), p=2, dim=1)
        
        pos_scores = model.predict_links(
            z_pos, pos_node_idx, pos_edge_idx, len(pos_edges), heuristic_scores=pos_heuristic
        )
        neg_scores = model.predict_links(
            z_neg, neg_node_idx, neg_edge_idx, len(neg_edges), heuristic_scores=neg_heuristic
        )
        
        labels_pos = torch.ones_like(pos_scores)
        labels_neg = torch.zeros_like(neg_scores)
        
        loss = criterion(pos_scores, labels_pos) + criterion(neg_scores, labels_neg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.memory_module.memory.detach_()
        
        # Parallel GPU heuristic memory update
        heuristic_engine.update(pos_edges)

    mean_ap = np.mean(ap_scores)
    mean_auc = np.mean(auc_scores)

    print("\n================ FINAL PREDICTION METRICS ================")
    print(f"Mean Average Precision (AP): {mean_ap:.4f}")
    print(f"Mean ROC-AUC: {mean_auc:.4f}")
    print("==========================================================")
    
    model_name = "temphyper_dec"
    output_filename = f"{model_name}_{dataset}_metrics.txt"
    with open(output_filename, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Mean Average Precision (AP): {mean_ap:.4f}\n")
        f.write(f"Mean ROC-AUC: {mean_auc:.4f}\n")

    model_save_path = os.path.join(PROJECT_ROOT, "temphyper_link_pred.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n-> Pre-trained model weights saved to: {model_save_path}")
    
    print("\nGenerating publication visuals...")
    generate_visuals(tracked_steps, ap_scores, auc_scores, df_labels, model)


def generate_visuals(steps, ap, auc, df_labels, model):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # ---------------------------------------------------------
    # FIGURE 1: The Learning Curve 
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    df_metrics = pd.DataFrame({'Step': steps, 'AP': ap, 'AUC': auc})
    df_metrics['AP_Smooth'] = df_metrics['AP'].rolling(window=15, min_periods=1).mean()
    df_metrics['AUC_Smooth'] = df_metrics['AUC'].rolling(window=15, min_periods=1).mean()
    
    plt.plot(df_metrics['Step'], df_metrics['AP_Smooth'], label='Average Precision (AP)', color='#1f77b4', linewidth=2.5)
    plt.plot(df_metrics['Step'], df_metrics['AUC_Smooth'], label='ROC-AUC', color='#ff7f0e', linewidth=2.5)
    plt.plot(df_metrics['Step'], df_metrics['AP'], color='#1f77b4', alpha=0.15)
    plt.plot(df_metrics['Step'], df_metrics['AUC'], color='#ff7f0e', alpha=0.15)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Random Baseline (0.5)')
    
    plt.title("TempHyper Stability Under Concept Drift", fontweight='bold', pad=15)
    plt.xlabel("Temporal Step")
    plt.ylabel("Predictive Score")
    plt.ylim(0.4, 1.05)
    plt.legend(loc='lower right', frameon=True)
    
    fig1_path = os.path.join(PROJECT_ROOT, "learning_curve.pdf")
    plt.savefig(fig1_path, format='pdf', bbox_inches='tight')
    print(f"-> Saved Publication-Grade Learning Curve to: {fig1_path}")
    plt.close()

    # ---------------------------------------------------------
    # FIGURE 2: t-SNE Latent Space 
    # ---------------------------------------------------------
    print("\n--- Generating t-SNE Latent Projection ---")
    plt.figure(figsize=(8, 8))
    
    z_active = (model.memory_module.get_memory() + model.node_emb.weight).detach().cpu().numpy()
    
    node_to_cluster = df_labels.groupby('node_id')['cluster'].last().to_dict()
    true_labels = np.array([node_to_cluster.get(i, -1) for i in range(z_active.shape[0])])
    
    nan_mask = ~np.isnan(z_active).any(axis=1)
    z_active = z_active[nan_mask]
    true_labels = true_labels[nan_mask]
    
    valid_mask = true_labels != -1
    z_active = z_active[valid_mask]
    true_labels = true_labels[valid_mask]
    
    top_departments = pd.Series(true_labels).value_counts().nlargest(7).index.tolist()
    mask = np.isin(true_labels, top_departments)
    z_filtered = z_active[mask]
    labels_filtered = true_labels[mask]
    
    if z_filtered.shape[0] > 10:
        tsne = TSNE(n_components=2, perplexity=min(30, z_filtered.shape[0]-1), random_state=42)
        z_2d = tsne.fit_transform(z_filtered)
        
        sns.scatterplot(
            x=z_2d[:, 0], y=z_2d[:, 1], 
            hue=labels_filtered, 
            palette="Set2", 
            s=100, alpha=0.85, edgecolor='w', linewidth=0.5
        )
        
        plt.title("Topological Semantic Mismatch (Latent Space)", fontweight='bold', pad=15)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(title="Ground Truth Dept", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig2_path = os.path.join(PROJECT_ROOT, "tsne_latent_space.pdf")
        plt.savefig(fig2_path, format='pdf', bbox_inches='tight')
        print(f"-> Saved Publication-Grade t-SNE Visualization to: {fig2_path}")
    else:
        print("-> Error: Not enough valid ground-truth nodes to plot.")
    plt.close()


if __name__ == "__main__":
    train_link_prediction(dataset="email")
