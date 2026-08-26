import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from models.temphyper_dec import TempHyperDEC

def target_distribution(q):
    """Calculates the sharpened target distribution P for DEC."""
    weight = (q ** 2) / torch.sum(q, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def run_clustering_finetune(dataset="email"):
    print("--- Loading Data & Pre-trained Model for DEC Fine-Tuning ---")
    
    # 1. Load Data to map true labels (for evaluation)
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    df_edges = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_edges.pkl"))
    df_labels = pd.read_pickle(os.path.join(DATA_RAW, f"{dataset}_labels.pkl"))

    all_unique_nodes = set()
    for edges in df_edges['hyperedge_nodes']:
        all_unique_nodes.update(edges)
    all_unique_nodes.update(df_labels['node_id'].unique())
    num_nodes = len(all_unique_nodes)
    
    # Define hyperparams (MUST match the pre-trained model exactly)
    MEMORY_DIM = 128
    TIME_DIM = 32
    HIDDEN_DIM = 64
    N_CLUSTERS = 6  # Targeting the top distinct departments
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Instantiate and Load the Model
    model = TempHyperDEC(num_nodes, MEMORY_DIM, TIME_DIM, HIDDEN_DIM, n_clusters=N_CLUSTERS, use_attention=True).to(device)
    model_path = os.path.join(PROJECT_ROOT, "temphyper_link_pred.pth")

    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Run link prediction first.")
        return
        
    # Load the dictionary of saved weights
    pretrained_dict = torch.load(model_path, map_location=device)
    
    # Strip out the old cluster centers to avoid the shape mismatch
    if 'cluster_centers' in pretrained_dict:
        del pretrained_dict['cluster_centers']
        
    # Load the remaining backbone weights with strict=False
    model.load_state_dict(pretrained_dict, strict=False)
    print("-> Successfully loaded pre-trained TempHyper backbone.")
    
    # if not os.path.exists(model_path):
    #     print(f"Error: Could not find {model_path}. Run link prediction first.")
    #     return
        
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # print("-> Successfully loaded pre-trained TempHyper backbone.")

    # 3. Extract the Frozen Representations
    model.eval()
    with torch.no_grad():
        # Z is our frozen topological memory
        z_frozen = (model.memory_module.get_memory() + model.node_emb.weight).detach()
    
    # 4. Freeze Backbone & Unfreeze Clustering Head
    for param in model.parameters():
        param.requires_grad = False
    model.cluster_centers.requires_grad = True
    
    # Initialize Cluster Centers using K-Means on the frozen embeddings
    print("-> Initializing cluster centers via K-Means...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42)
    kmeans.fit(z_frozen.cpu().numpy())
    model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    
    # 5. Prepare the clustering optimizer and evaluation labels
    cluster_optimizer = optim.Adam([model.cluster_centers], lr=0.01)
    
    node_to_cluster = df_labels.groupby('node_id')['cluster'].last().to_dict()
    true_labels = np.array([node_to_cluster.get(i, -1) for i in range(num_nodes)])
    valid_mask = true_labels != -1
    
    # Filter only to the top N_CLUSTERS for cleaner evaluation
    top_departments = pd.Series(true_labels[valid_mask]).value_counts().nlargest(N_CLUSTERS).index.tolist()
    eval_mask = np.isin(true_labels, top_departments) & valid_mask
    
    print("\n--- Starting Deep Embedded Clustering (DEC) Fine-Tuning ---")
    epochs = 100
    
    for epoch in range(epochs):
        model.train()
        cluster_optimizer.zero_grad()
        
        # Calculate soft assignments (Q) distance from nodes to cluster centers
        dist = torch.sum((z_frozen.unsqueeze(1) - model.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / model.alpha)
        q = q ** ((model.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        
        # Calculate target distribution (P)
        p = target_distribution(q).detach()
        
        # KL Divergence Loss
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        
        loss.backward()
        cluster_optimizer.step()
        
        # Periodic Evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = torch.argmax(q, dim=1).cpu().numpy()
                
                # Evaluate only on valid nodes belonging to top target classes
                y_true = true_labels[eval_mask]
                y_pred = preds[eval_mask]
                
                ari = adjusted_rand_score(y_true, y_pred)
                nmi = normalized_mutual_info_score(y_true, y_pred)
                
            print(f"Epoch {epoch:03d} | KL Loss: {loss.item():.4f} | ARI: {ari:.4f} | NMI: {nmi:.4f}")

    print("\n================ FINAL CLUSTERING METRICS ================")
    print(f"Final Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Final Normalized Mutual Info (NMI): {nmi:.4f}")
    print("==========================================================")

if __name__ == "__main__":
    run_clustering_finetune(dataset="email")