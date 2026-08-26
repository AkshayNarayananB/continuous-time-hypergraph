import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SYNTHETIC = os.path.join(PROJECT_ROOT, "data", "synthetic")
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from models.temphyper_dec import TempHyperDEC
from losses import target_distribution, dec_kl_loss
from utils import evaluate_clustering

def prepare_hyperedge_batch(batch_edges, device):
    node_idx, edge_idx = [], []
    for e_id, edge in enumerate(batch_edges):
        for node in edge:
            node_idx.append(node)
            edge_idx.append(e_id)
    return torch.tensor(node_idx, dtype=torch.long, device=device), \
           torch.tensor(edge_idx, dtype=torch.long, device=device)

def train_temphyper_dec(dataset="email"):
    print(f"--- Loading {dataset.upper()} Dataset ---")
    
    if dataset == "synthetic":
        df_edges = pd.read_pickle(os.path.join(DATA_SYNTHETIC, "synthetic_edges.pkl"))
        df_labels = pd.read_pickle(os.path.join(DATA_SYNTHETIC, "synthetic_labels.pkl"))
    elif dataset == "email":
        df_edges = pd.read_pickle(os.path.join(DATA_RAW, "email_edges.pkl"))
        df_labels = pd.read_pickle(os.path.join(DATA_RAW, "email_labels.pkl"))
    elif dataset == "math":
        df_edges = pd.read_pickle(os.path.join(DATA_RAW, "math_edges.pkl"))
        df_labels = pd.read_pickle(os.path.join(DATA_RAW, "math_labels.pkl"))
    else:
        raise ValueError("Invalid dataset.")

    # --- Contiguous ID Mapping ---
    all_unique_nodes = set()
    for edges in df_edges['hyperedge_nodes']:
        all_unique_nodes.update(edges)
    all_unique_nodes.update(df_labels['node_id'].unique())
    node_mapping = {raw_id: new_id for new_id, raw_id in enumerate(sorted(all_unique_nodes))}
    df_edges['hyperedge_nodes'] = df_edges['hyperedge_nodes'].apply(lambda edge: tuple(node_mapping[n] for n in edge))
    df_labels['node_id'] = df_labels['node_id'].map(node_mapping)
        
    num_nodes = len(node_mapping)
    num_clusters = df_labels['cluster'].nunique()
    time_steps = sorted(df_edges['time_step'].unique())
    
    # --- Tuned Hyperparameters ---
    MEMORY_DIM = 128
    TIME_DIM = 32
    HIDDEN_DIM = 64
    LR = 0.005  
    TAU = 0.5             
    EMA_MOMENTUM = 0.9    
    LAMBDA_DEC = 0.5      
    
    WARMUP_EPOCHS = 30
    EPOCHS_PER_STEP = 2 
    DEC_START_T = 30 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Nodes: {num_nodes} | Clusters: {num_clusters} | Time Steps: {len(time_steps)}")
    
    model = TempHyperDEC(num_nodes, MEMORY_DIM, TIME_DIM, HIDDEN_DIM, num_clusters, alpha=0.2, use_attention=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    print("\n--- Starting Execution with Rolling Window Probing ---")
    nmi_scores, ari_scores, micro_f1_scores = [], [], []
    kmeans_initialized = False
    
    # TRACKER: When did each node last interact?
    node_last_active = np.zeros(num_nodes)

    for t_idx, t in enumerate(time_steps):
        current_edges = df_edges[df_edges['time_step'] == t]['hyperedge_nodes'].tolist()
        if not current_edges:
            continue
            
        # Update the last active time for these nodes
        for edge in current_edges:
            for node in edge:
                node_last_active[node] = t_idx
                
        current_t_tensor = torch.tensor([t_idx], dtype=torch.float32, device=device)
        node_indices, hyperedge_indices = prepare_hyperedge_batch(current_edges, device)
        gt_labels = df_labels[df_labels['time_step'] == t].sort_values('node_id')['cluster'].values
        
        epochs_this_step = WARMUP_EPOCHS if t_idx == 0 else EPOCHS_PER_STEP
        
        model.train()
        for epoch in range(epochs_this_step):
            optimizer.zero_grad()
            
            z, q = model(node_indices, hyperedge_indices, current_t_tensor)
            
            # --- InfoNCE Contrastive Loss ---
            gathered_z = z[node_indices]
            idx_edge = hyperedge_indices.unsqueeze(1).expand_as(gathered_z)
            
            num_active_edges = int(hyperedge_indices.max().item()) + 1
            edge_centers = torch.zeros(num_active_edges, z.size(1), device=device)
            edge_centers.scatter_reduce_(0, idx_edge, gathered_z, reduce='mean', include_self=False)
            
            pos_scores = torch.sum(gathered_z * edge_centers[hyperedge_indices], dim=1)
            random_edges = torch.randint(0, num_active_edges, (len(node_indices),), device=device)
            neg_scores = torch.sum(gathered_z * edge_centers[random_edges], dim=1)
            
            logits = torch.stack([pos_scores, neg_scores], dim=1) / TAU
            labels = torch.zeros(len(node_indices), dtype=torch.long, device=device)
            loss_struct = F.cross_entropy(logits, labels)

            loss_dec = torch.tensor(0.0, device=device)

            if t_idx >= DEC_START_T and epoch == 0 and not kmeans_initialized:
                print(f"\n>>> Graph populated. Initializing Centroids at Step {t_idx} <<<")
                kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
                all_h = model.memory_module.get_memory() + model.node_emb.weight
                all_z = F.normalize(model.hypergraph_layer(all_h, node_indices, hyperedge_indices), p=2, dim=1)
                
                kmeans.fit(all_z.detach().cpu().numpy())
                centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
                model.cluster_centers.data = F.normalize(centers, p=2, dim=1)
                kmeans_initialized = True
            
            if kmeans_initialized:
                dist = torch.sum((z.unsqueeze(1) - model.cluster_centers.unsqueeze(0)) ** 2, dim=2)
                q_active = 1.0 / (1.0 + dist / model.alpha)
                q_active = (q_active ** ((model.alpha + 1.0) / 2.0))
                q_active = (q_active.t() / torch.sum(q_active, dim=1)).t()
                
                p = target_distribution(q_active).detach()
                loss_dec = dec_kl_loss(q_active, p)
                
                model.update_centroids_ema(z.detach(), q_active.detach(), momentum=EMA_MOMENTUM)

            total_loss = loss_struct + (LAMBDA_DEC * loss_dec)
            total_loss.backward()
            optimizer.step()
            
            model.memory_module.memory.detach_()
        
        # --- Evaluation & Linear Probing (Rolling Window) ---
        if t_idx % 10 == 0 or t_idx == len(time_steps) - 1:
            model.eval()
            with torch.no_grad():
                if kmeans_initialized:
                    all_h = model.memory_module.get_memory() + model.node_emb.weight
                    all_z = F.normalize(model.hypergraph_layer(all_h, node_indices, hyperedge_indices), p=2, dim=1)
                    
                    # THE FIX: Grab users active in the last 30 time steps (Rolling Window)
                    recent_active_nodes = np.where((t_idx - node_last_active) <= 30)[0]
                    
                    # Ensure we have a statistically significant sample to test on
                    if len(recent_active_nodes) < 100:
                        continue 
                        
                    dist_all = torch.sum((all_z.unsqueeze(1) - model.cluster_centers.unsqueeze(0)) ** 2, dim=2)
                    q_all = 1.0 / (1.0 + dist_all / model.alpha)
                    
                    pred_labels_active = torch.argmax(q_all, dim=1).cpu().numpy()[recent_active_nodes]
                    gt_labels_active = gt_labels[recent_active_nodes]
                    z_active = all_z.cpu().numpy()[recent_active_nodes]
                    
                    metrics = evaluate_clustering(z_active, gt_labels_active, pred_labels_active)
                    
                    nmi_scores.append(metrics['NMI'])
                    ari_scores.append(metrics['ARI'])
                    
                    # --- THE LINEAR PROBE (Rolling Window) ---
                    active_classes = len(np.unique(gt_labels_active))
                    if active_classes > 5:  # Ensure enough classes exist to make probing valid
                        X_train, X_test, y_train, y_test = train_test_split(
                            z_active, gt_labels_active, test_size=0.5, random_state=42
                        )
                        clf = LogisticRegression(max_iter=500, class_weight='balanced')
                        clf.fit(X_train, y_train)
                        preds_probe = clf.predict(X_test)
                        
                        micro_f1 = f1_score(y_test, preds_probe, average='micro')
                        micro_f1_scores.append(micro_f1)
                    else:
                        micro_f1 = 0.0
                        
                    print(f"Step {t_idx:03d} | Active Users: {len(recent_active_nodes):03d} | Loss: {total_loss.item():.4f} | NMI: {metrics['NMI']:.4f} | ARI: {metrics['ARI']:.4f} | Probe Micro-F1: {micro_f1:.4f}")
                else:
                    print(f"Step {t_idx:03d} | Warming up... Loss: {total_loss.item():.4f}")

    print("\n================ FINAL ARCHITECTURE RESULTS ================")
    if nmi_scores:
        print(f"Average NMI: {np.mean(nmi_scores):.4f}")
        print(f"Average ARI: {np.mean(ari_scores):.4f}")
        print(f"Average Linear Probe Micro-F1: {np.mean(micro_f1_scores):.4f}")
    print("============================================================")

if __name__ == "__main__":
    train_temphyper_dec(dataset="math")