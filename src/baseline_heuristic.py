import os
import sys
import torch
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


def prepare_hyperedge_tensors(hyperedges, device):
    """
    Separates hyperedges into singleton nodes (len==1) and constituent pairs (len>=2)
    and produces flat index tensors and segment maps for parallel GPU execution.
    """
    pair_u, pair_v, pair_seg_ids = [], [], []
    single_nodes, single_seg_ids = [], []
    pair_counts = np.zeros(len(hyperedges), dtype=np.float32)

    for edge_idx, edge in enumerate(hyperedges):
        if len(edge) >= 2:
            pairs = list(combinations(edge, 2))
            u_nodes, v_nodes = zip(*pairs)
            pair_u.extend(u_nodes)
            pair_v.extend(v_nodes)
            pair_seg_ids.extend([edge_idx] * len(pairs))
            pair_counts[edge_idx] = float(len(pairs))
        elif len(edge) == 1:
            single_nodes.append(edge[0])
            single_seg_ids.append(edge_idx)

    # Convert to GPU tensors
    pair_info = None
    if pair_u:
        pair_info = (
            torch.tensor(pair_u, dtype=torch.long, device=device),
            torch.tensor(pair_v, dtype=torch.long, device=device),
            torch.tensor(pair_seg_ids, dtype=torch.long, device=device),
        )

    single_info = None
    if single_nodes:
        single_info = (
            torch.tensor(single_nodes, dtype=torch.long, device=device),
            torch.tensor(single_seg_ids, dtype=torch.long, device=device),
        )

    counts = torch.from_numpy(pair_counts).to(device)
    return pair_info, single_info, counts, len(hyperedges)


class GPUHeuristicEngine:
    """
    GPU-accelerated state tracker for node frequencies and pairwise co-occurrences.
    """
    def __init__(self, num_nodes, device):
        self.num_nodes = num_nodes
        self.device = device
        self.node_frequency = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.co_occurrence_matrix = torch.sparse_coo_tensor(
            size=(num_nodes, num_nodes), dtype=torch.float32, device=device
        )
        self.cached_co_dense = None

    def update(self, pos_edges):
        """Updates frequency and co-occurrence counts using batch GPU operations."""
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
            # Accumulate and coalesce sparse matrix
            self.co_occurrence_matrix = (self.co_occurrence_matrix + new_co).coalesce()
            self.cached_co_dense = None

    def _get_dense_view(self):
        if self.cached_co_dense is None:
            self.cached_co_dense = self.co_occurrence_matrix.to_dense()
        return self.cached_co_dense

    def score(self, edge_data):
        """Computes average pairwise co-occurrence scores completely in parallel on GPU."""
        if edge_data is None:
            return torch.zeros(0, device=self.device)

        pair_info, single_info, counts, total_edges = edge_data
        scores = torch.zeros(total_edges, dtype=torch.float32, device=self.device)

        # 1. Score pair combinations in parallel via direct 2D indexed access
        if pair_info is not None:
            u, v, seg_ids = pair_info
            dense_co = self._get_dense_view()
            pair_values = dense_co[u, v]
            summed_scores = scatter(pair_values, seg_ids, dim=0, dim_size=total_edges, reduce='sum')
            # Compute average score per edge
            scores = scores + torch.where(counts > 0, summed_scores / torch.clamp(counts, min=1.0), 0.0)

        # 2. Score singletons directly from node frequency tensor
        if single_info is not None:
            nodes, seg_ids = single_info
            singleton_values = self.node_frequency[nodes]
            scores[seg_ids] = singleton_values

        return scores


def run_heuristic_baseline(dataset="mathoverflow"):
    print(f"--- Running OPTIMIZED GPU HEURISTIC Baseline on {dataset.upper()} ---")

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
    print(f"Nodes: {num_nodes} | Time Steps: {len(time_steps)} | Device: {device}")

    engine = GPUHeuristicEngine(num_nodes, device)
    seen_hyperedges = set()

    ap_scores, auc_scores = [], []
    novel_ap_scores, novel_auc_scores = [], []

    # Pre-aggregate timesteps to bypass repeated Pandas filtering
    time_grouped_edges = {t: group['hyperedge_nodes'].tolist() for t, group in df_edges.groupby('time_step')}

    for t_idx, t in enumerate(time_steps):
        pos_edges = time_grouped_edges.get(t, [])
        if not pos_edges:
            continue

        neg_edges = generate_negative_hyperedges_batch(pos_edges, num_nodes)

        novel_pos_edges = []
        novel_neg_edges = []
        for p_edge, n_edge in zip(pos_edges, neg_edges):
            if p_edge not in seen_hyperedges:
                novel_pos_edges.append(p_edge)
                novel_neg_edges.append(n_edge)

        pos_data = prepare_hyperedge_tensors(pos_edges, device)
        neg_data = prepare_hyperedge_tensors(neg_edges, device)

        # ==========================================
        # PHASE 1: PREDICT (GPU-Accelerated)
        # ==========================================
        if t_idx > 10:
            with torch.inference_mode():
                pos_scores = engine.score(pos_data)
                neg_scores = engine.score(neg_data)

                # Append small uniform noise on GPU to break ties without CPU sync overhead
                pos_noise = torch.rand_like(pos_scores) * 1e-5
                neg_noise = torch.rand_like(neg_scores) * 1e-5

                all_scores = torch.cat([pos_scores + pos_noise, neg_scores + neg_noise]).cpu().numpy()
                y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

                ap_scores.append(average_precision_score(y_true, all_scores))
                auc_scores.append(roc_auc_score(y_true, all_scores))

                # --- NOVEL Edges Evaluation ---
                if novel_pos_edges and novel_neg_edges:
                    novel_pos_data = prepare_hyperedge_tensors(novel_pos_edges, device)
                    novel_neg_data = prepare_hyperedge_tensors(novel_neg_edges, device)

                    n_pos = engine.score(novel_pos_data)
                    n_neg = engine.score(novel_neg_data)

                    n_pos_noise = torch.rand_like(n_pos) * 1e-5
                    n_neg_noise = torch.rand_like(n_neg) * 1e-5

                    n_scores = torch.cat([n_pos + n_pos_noise, n_neg + n_neg_noise]).cpu().numpy()
                    n_y_true = np.concatenate([np.ones(len(n_pos)), np.zeros(len(n_neg))])

                    try:
                        novel_ap_scores.append(average_precision_score(n_y_true, n_scores))
                        novel_auc_scores.append(roc_auc_score(n_y_true, n_scores))
                    except ValueError:
                        pass

                if t_idx % 50 == 0:
                    novel_print = f"{novel_auc_scores[-1]:.4f}" if novel_auc_scores else "N/A"
                    print(f"Step {t_idx:03d} | ALL AUC: {auc_scores[-1]:.4f} | NOVEL AUC: {novel_print} (Count: {len(novel_pos_edges)})")

        # ==========================================
        # PHASE 2: TRAIN (Update GPU Frequency Banks)
        # ==========================================
        seen_hyperedges.update(pos_edges)
        engine.update(pos_edges)

    mean_ap = np.mean(ap_scores)
    mean_auc = np.mean(auc_scores)
    mean_novel_ap = np.mean(novel_ap_scores) if novel_ap_scores else float('nan')
    mean_novel_auc = np.mean(novel_auc_scores) if novel_auc_scores else float('nan')

    print("\n================ HEURISTIC METRICS (ALL EDGES) ================")
    print(f"Mean Average Precision (AP): {mean_ap:.4f}")
    print(f"Mean ROC-AUC: {mean_auc:.4f}")

    print("\n================ HEURISTIC METRICS (NOVEL EDGES ONLY) =========")
    print(f"Mean Average Precision (AP): {mean_novel_ap:.4f}")
    print(f"Mean ROC-AUC: {mean_novel_auc:.4f}")
    print("===============================================================")

    model_name = "heuristic"
    output_filename = f"{model_name}_{dataset}_metrics.txt"
    with open(output_filename, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Mean Average Precision (AP) [ALL]: {mean_ap:.4f}\n")
        f.write(f"Mean ROC-AUC [ALL]: {mean_auc:.4f}\n")
        f.write(f"Mean Average Precision (AP) [NOVEL]: {mean_novel_ap:.4f}\n")
        f.write(f"Mean ROC-AUC [NOVEL]: {mean_novel_auc:.4f}\n")


if __name__ == "__main__":
    run_heuristic_baseline(dataset="email")
