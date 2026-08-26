import os
import sys
import time
import json
import random
import warnings
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from models.temphyper_dec import TempHyperDEC

# ==========================================
# HELPER FUNCTIONS & DATA UTILITIES
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_hyperedge_batch(batch_edges, device):
    node_idx, edge_idx = [], []
    for e_id, edge in enumerate(batch_edges):
        for node in edge:
            node_idx.append(node)
            edge_idx.append(e_id)
    return (
        torch.tensor(node_idx, dtype=torch.long, device=device),
        torch.tensor(edge_idx, dtype=torch.long, device=device)
    )

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

def load_dataset(dataset_name):
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    edges_path = os.path.join(DATA_RAW, f"{dataset_name}_edges.pkl")
    labels_path = os.path.join(DATA_RAW, f"{dataset_name}_labels.pkl")

    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing {edges_path}. Run preprocessing first.")

    df_edges = pd.read_pickle(edges_path)
    df_labels = pd.read_pickle(labels_path)

    all_unique_nodes = set()
    for edges in df_edges['hyperedge_nodes']:
        all_unique_nodes.update(edges)
    all_unique_nodes.update(df_labels['node_id'].unique())
    node_mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted(all_unique_nodes))}

    df_edges['hyperedge_nodes'] = df_edges['hyperedge_nodes'].apply(
        lambda edge: tuple(sorted(node_mapping[n] for n in edge))
    )
    df_labels['node_id'] = df_labels['node_id'].map(node_mapping)

    return df_edges, df_labels, len(node_mapping)

# ==========================================
# PAIRWISE CLIQUE CT-RNN (FOR BENCHMARKING)
# ==========================================

class PairwiseCTRNNBenchmark(nn.Module):
    """Pairwise CT-RNN model that expands hyperedges into O(N^2) cliques."""
    def __init__(self, num_nodes, emb_dim=64):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )
        self.register_buffer("memory", torch.zeros(num_nodes, emb_dim))

    def forward_pairwise(self, u_idx, v_idx):
        h = self.memory + self.node_emb.weight
        u_emb = h[u_idx]
        v_emb = h[v_idx]
        cat_feat = torch.cat([u_emb, v_emb], dim=-1)
        return self.mlp(cat_feat).squeeze(-1)

# ==========================================
# 1. RUNTIME & COMPUTATIONAL EFFICIENCY BENCHMARK
# ==========================================

def run_runtime_benchmark(dataset_name, df_edges, num_nodes, device):
    print(f"\n>>> Running Computational Efficiency Benchmark on [{dataset_name.upper()}]...")

    time_steps = sorted(df_edges['time_step'].unique())
    sample_steps = time_steps[:min(50, len(time_steps))]

    # 1. Setup TempHyper Model
    temphyper = TempHyperDEC(num_nodes, memory_dim=64, time_dim=32, hidden_dim=32, n_clusters=2, use_attention=True).to(device)
    temphyper_opt = optim.Adam(temphyper.parameters(), lr=0.003)
    criterion = nn.BCEWithLogitsLoss()

    # 2. Setup Pairwise CT-RNN Model
    ctrnn = PairwiseCTRNNBenchmark(num_nodes, emb_dim=64).to(device)
    ctrnn_opt = optim.Adam(ctrnn.parameters(), lr=0.003)

    # CUDA Timers
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    # Benchmark TempHyper (Native O(N) Hyperedge)
    temphyper_step_times = []
    for t in sample_steps:
        pos_edges = df_edges[df_edges['time_step'] == t]['hyperedge_nodes'].tolist()
        if not pos_edges:
            continue
        neg_edges = generate_negative_hyperedges(pos_edges, num_nodes)
        current_t = torch.tensor([t], dtype=torch.float32, device=device)

        pos_node_idx, pos_edge_idx = prepare_hyperedge_batch(pos_edges, device)
        neg_node_idx, neg_edge_idx = prepare_hyperedge_batch(neg_edges, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event.record()
        else:
            t0 = time.perf_counter()

        temphyper_opt.zero_grad()
        z_pos, _ = temphyper(pos_node_idx, pos_edge_idx, current_t)
        h = temphyper.memory_module.get_memory() + temphyper.node_emb.weight
        z_neg = F.normalize(temphyper.hypergraph_layer(h, neg_node_idx, neg_edge_idx), p=2, dim=1)

        pos_scores = temphyper.predict_links(z_pos, pos_node_idx, pos_edge_idx, len(pos_edges))
        neg_scores = temphyper.predict_links(z_neg, neg_node_idx, neg_edge_idx, len(neg_edges))

        loss = criterion(pos_scores, torch.ones_like(pos_scores)) + criterion(neg_scores, torch.zeros_like(neg_scores))
        loss.backward()
        temphyper_opt.step()
        temphyper.memory_module.memory.detach_()

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000

        temphyper_step_times.append(elapsed_ms)

    # Benchmark CT-RNN (O(N^2) Clique Expansion)
    ctrnn_step_times = []
    for t in sample_steps:
        pos_edges = df_edges[df_edges['time_step'] == t]['hyperedge_nodes'].tolist()
        if not pos_edges:
            continue

        # Flatten hyperedges into pairwise cliques
        u_pos, v_pos = [], []
        for e in pos_edges:
            if len(e) >= 2:
                for u, v in combinations(e, 2):
                    u_pos.append(u)
                    v_pos.append(v)
            elif len(e) == 1:
                u_pos.append(e[0])
                v_pos.append(e[0])

        if not u_pos:
            continue

        u_neg = [random.randint(0, num_nodes - 1) for _ in range(len(u_pos))]
        v_neg = [random.randint(0, num_nodes - 1) for _ in range(len(v_pos))]

        u_pos_t = torch.tensor(u_pos, dtype=torch.long, device=device)
        v_pos_t = torch.tensor(v_pos, dtype=torch.long, device=device)
        u_neg_t = torch.tensor(u_neg, dtype=torch.long, device=device)
        v_neg_t = torch.tensor(v_neg, dtype=torch.long, device=device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event.record()
        else:
            t0 = time.perf_counter()

        ctrnn_opt.zero_grad()
        pos_scores = ctrnn.forward_pairwise(u_pos_t, v_pos_t)
        neg_scores = ctrnn.forward_pairwise(u_neg_t, v_neg_t)

        loss = criterion(pos_scores, torch.ones_like(pos_scores)) + criterion(neg_scores, torch.zeros_like(neg_scores))
        loss.backward()
        ctrnn_opt.step()
        ctrnn.memory.detach_()

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000

        ctrnn_step_times.append(elapsed_ms)

    temphyper_avg_ms = np.mean(temphyper_step_times)
    ctrnn_avg_ms = np.mean(ctrnn_step_times)
    speedup = ctrnn_avg_ms / max(temphyper_avg_ms, 1e-6)

    print(f"-> TempHyper Avg Step Latency: {temphyper_avg_ms:.2f} ms")
    print(f"-> CT-RNN (Clique) Avg Step Latency: {ctrnn_avg_ms:.2f} ms")
    print(f"-> TempHyper Acceleration: {speedup:.2f}x faster")

    return {
        "dataset": dataset_name,
        "temphyper_avg_step_ms": round(temphyper_avg_ms, 3),
        "ctrnn_clique_avg_step_ms": round(ctrnn_avg_ms, 3),
        "speedup_factor": round(speedup, 2)
    }

# ==========================================
# 2. ABLATION STUDY EXPERIMENTS
# ==========================================

def run_ablation_variant(df_edges, num_nodes, device, variant_name="full"):
    """
    Variants:
    - 'full': TempHyper (Memory + HyperGAT + Feature Injection)
    - 'no_feature_injection': Pure Latent (Memory + HyperGAT, no heuristic)
    - 'no_temporal_memory': Static Hypergraph (HyperGAT + Feature Injection, memory frozen)
    - 'no_attention': Mean-Pooling Hypergraph + Memory + Feature Injection
    """
    set_seed(42)
    time_steps = sorted(df_edges['time_step'].unique())

    use_att = False if variant_name == "no_attention" else True
    use_mem = False if variant_name == "no_temporal_memory" else True
    use_feat_inj = False if variant_name == "no_feature_injection" else True

    model = TempHyperDEC(num_nodes, memory_dim=64, time_dim=32, hidden_dim=32, n_clusters=2, use_attention=use_att).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    co_occurrence_tracker = defaultdict(int)
    node_frequency = defaultdict(int)
    seen_hyperedges = set()

    all_auc_scores, novel_auc_scores = [], []
    all_ap_scores = []

    for t_idx, t in enumerate(time_steps):
        pos_edges = df_edges[df_edges['time_step'] == t]['hyperedge_nodes'].tolist()
        if not pos_edges:
            continue

        current_t = torch.tensor([t_idx], dtype=torch.float32, device=device)
        neg_edges = generate_negative_hyperedges(pos_edges, num_nodes)

        pos_node_idx, pos_edge_idx = prepare_hyperedge_batch(pos_edges, device)
        neg_node_idx, neg_edge_idx = prepare_hyperedge_batch(neg_edges, device)

        pos_heuristic = torch.tensor([score_heuristic(e, co_occurrence_tracker, node_frequency) for e in pos_edges], dtype=torch.float32, device=device) if use_feat_inj else None
        neg_heuristic = torch.tensor([score_heuristic(e, co_occurrence_tracker, node_frequency) for e in neg_edges], dtype=torch.float32, device=device) if use_feat_inj else None

        # Predict Step
        if t_idx > 10:
            model.eval()
            with torch.no_grad():
                h = (model.memory_module.get_memory() if use_mem else 0) + model.node_emb.weight
                z_pos_eval = F.normalize(model.hypergraph_layer(h, pos_node_idx, pos_edge_idx), p=2, dim=1)
                z_neg_eval = F.normalize(model.hypergraph_layer(h, neg_node_idx, neg_edge_idx), p=2, dim=1)

                pos_scores_eval = model.predict_links(z_pos_eval, pos_node_idx, pos_edge_idx, len(pos_edges), heuristic_scores=pos_heuristic)
                neg_scores_eval = model.predict_links(z_neg_eval, neg_node_idx, neg_edge_idx, len(neg_edges), heuristic_scores=neg_heuristic)

                y_true = np.concatenate([np.ones(len(pos_scores_eval)), np.zeros(len(neg_scores_eval))])
                y_scores = torch.cat([pos_scores_eval, neg_scores_eval]).cpu().numpy()

                if not np.isnan(y_scores).any():
                    all_auc_scores.append(roc_auc_score(y_true, y_scores))
                    all_ap_scores.append(average_precision_score(y_true, y_scores))

                    # Track inductive / novel edges
                    novel_mask_pos = [e not in seen_hyperedges for e in pos_edges]
                    novel_pos_scores = pos_scores_eval[torch.tensor(novel_mask_pos, device=device)]
                    if len(novel_pos_scores) > 0:
                        y_true_nov = np.concatenate([np.ones(len(novel_pos_scores)), np.zeros(len(neg_scores_eval))])
                        y_scores_nov = torch.cat([novel_pos_scores, neg_scores_eval]).cpu().numpy()
                        novel_auc_scores.append(roc_auc_score(y_true_nov, y_scores_nov))

        # Train Step
        model.train()
        optimizer.zero_grad()

        if use_mem:
            z_pos, _ = model(pos_node_idx, pos_edge_idx, current_t)
            h_updated = model.memory_module.get_memory() + model.node_emb.weight
        else:
            h_updated = model.node_emb.weight
            z_pos = F.normalize(model.hypergraph_layer(h_updated, pos_node_idx, pos_edge_idx), p=2, dim=1)

        z_neg = F.normalize(model.hypergraph_layer(h_updated, neg_node_idx, neg_edge_idx), p=2, dim=1)

        pos_scores = model.predict_links(z_pos, pos_node_idx, pos_edge_idx, len(pos_edges), heuristic_scores=pos_heuristic)
        neg_scores = model.predict_links(z_neg, neg_node_idx, neg_edge_idx, len(neg_edges), heuristic_scores=neg_heuristic)

        loss = criterion(pos_scores, torch.ones_like(pos_scores)) + criterion(neg_scores, torch.zeros_like(neg_scores))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if use_mem:
            model.memory_module.memory.detach_()

        for edge in pos_edges:
            seen_hyperedges.add(edge)
            for node in edge:
                node_frequency[node] += 1
            if len(edge) >= 2:
                for u, v in combinations(edge, 2):
                    co_occurrence_tracker[tuple(sorted([u, v]))] += 1

    return {
        "ALL_AUC": round(float(np.mean(all_auc_scores)), 4),
        "NOVEL_AUC": round(float(np.mean(novel_auc_scores)), 4) if novel_auc_scores else 0.0,
        "ALL_AP": round(float(np.mean(all_ap_scores)), 4)
    }

def run_full_ablation_study(dataset_name, df_edges, num_nodes, device):
    print(f"\n>>> Running Ablation Study on [{dataset_name.upper()}]...")

    variants = [
        ("TempHyper (Full)", "full"),
        ("w/o Feature Injection", "no_feature_injection"),
        ("w/o Temporal Memory", "no_temporal_memory"),
        ("w/o Hypergraph Attention", "no_attention")
    ]

    results = {}
    for display_name, variant_key in variants:
        print(f"  -> Testing: {display_name}...")
        metrics = run_ablation_variant(df_edges, num_nodes, device, variant_name=variant_key)
        results[display_name] = metrics
        print(f"     ALL AUC: {metrics['ALL_AUC']:.4f} | NOVEL AUC: {metrics['NOVEL_AUC']:.4f} | AP: {metrics['ALL_AP']:.4f}")

    return results

# ==========================================
# MASTER RUNNER
# ==========================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"STARTING EFFICIENCY BENCHMARK & ABLATION PIPELINE")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    datasets = ["email", "mathoverflow"]
    runtime_records = []
    ablation_records = {}

    for ds in datasets:
        print(f"\n" + "#" * 60)
        print(f"PROCESSING DATASET: {ds.upper()}")
        print("#" * 60)

        df_edges, df_labels, num_nodes = load_dataset(ds)
        print(f"Loaded {ds}: {len(df_edges)} hyperedges, {num_nodes} unique nodes.")

        # 1. Runtime Benchmark
        rt_res = run_runtime_benchmark(ds, df_edges, num_nodes, device)
        runtime_records.append(rt_res)

        # 2. Ablation Study
        abl_res = run_full_ablation_study(ds, df_edges, num_nodes, device)
        ablation_records[ds] = abl_res

    # ==========================================
    # SAVE AND FORMAT OUTPUTS
    # ==========================================

    # 1. Write Efficiency Report
    eff_file = os.path.join(PROJECT_ROOT, "efficiency_benchmark_results.txt")
    with open(eff_file, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("COMPUTATIONAL RUNTIME BENCHMARK (TEMPHYPER VS CT-RNN)\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"{'Dataset':<15} | {'TempHyper (O(N))':<18} | {'CT-RNN (O(N^2))':<18} | {'Speedup':<10}\n")
        f.write("-" * 65 + "\n")
        for rec in runtime_records:
            f.write(f"{rec['dataset'].capitalize():<15} | {rec['temphyper_avg_step_ms']:>12.2f} ms/step | {rec['ctrnn_clique_avg_step_ms']:>12.2f} ms/step | {rec['speedup_factor']:>7.2f}x\n")
    print(f"\n[SUCCESS] Efficiency benchmark results saved to: {eff_file}")

    # 2. Write Ablation Study Report
    abl_file = os.path.join(PROJECT_ROOT, "ablation_study_results.txt")
    with open(abl_file, "w") as f:
        f.write("=" * 75 + "\n")
        f.write("TEMPHYPER COMPONENT ABLATION STUDY\n")
        f.write("=" * 75 + "\n\n")
        for ds in datasets:
            f.write(f"--- Dataset: {ds.upper()} ---\n")
            f.write(f"{'Model Configuration':<30} | {'ALL ROC-AUC':<12} | {'NOVEL ROC-AUC':<14} | {'ALL AP':<10}\n")
            f.write("-" * 75 + "\n")
            for model_name, metrics in ablation_records[ds].items():
                f.write(f"{model_name:<30} | {metrics['ALL_AUC']:>11.4f} | {metrics['NOVEL_AUC']:>13.4f} | {metrics['ALL_AP']:>8.4f}\n")
            f.write("\n")
    print(f"[SUCCESS] Ablation study results saved to: {abl_file}")

if __name__ == "__main__":
    main()
